/*
 * WISEE - recepteur CSI sans fil (ESP32-S3)
 *
 * Base : esp-csi/examples/get-started/csi_recv (Espressif, domaine public / CC0).
 * Modifications WISEE :
 *   - la carte se connecte en station (STA) au point d'acces WISEE (carte de secours,
 *     projet wisee_ap) sur le canal 11 HT40, avec une IP fixe 192.168.4.(200 + RX_ID) ;
 *   - chaque trame CSI de l'emetteur (MAC 1a:00:00:00:00:00) est mise en forme
 *     exactement comme la ligne CSI_DATA imprimee sur le port serie auparavant,
 *     puis envoyee en UDP au PC (port 5005) par une tache dediee (file FreeRTOS) ;
 *   - le PC se fait connaitre en envoyant "WISEE_HELLO" au port 5005 de la carte
 *     (record_multi.py --udp le fait automatiquement) : la carte envoie a l'adresse
 *     d'ou vient le dernier HELLO ;
 *   - plus de MAC forcee : chaque recepteur garde sa MAC d'usine (obligatoire pour
 *     se connecter a plusieurs au meme point d'acces) ;
 *   - le port serie n'affiche plus que des statistiques toutes les 5 s.
 *
 * A CHANGER AVANT DE FLASHER CHAQUE CARTE : RX_ID (1, 2, 3 ou 4).
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"
#include "lwip/sockets.h"

/* ======================= REGLAGES ======================= */
#define RX_ID                   4               /* 1, 2, 3 ou 4 : une valeur par carte */

#define WISEE_SSID              "WISEE_CSI"     /* identiques a wisee_ap */
#define WISEE_PASSWORD          "wisee2026"
#define WISEE_CHANNEL           11              /* meme canal que csi_send */
#define UDP_PORT                5005            /* port du PC (record_multi.py --udp 5005) */
/* ======================================================== */

#define CONFIG_FORCE_GAIN       1
#define CONFIG_GAIN_CONTROL     1

#define LINE_MAX_LEN            3072            /* une ligne CSI_DATA (384 valeurs) fait ~1,7 ko */
#define QUEUE_LEN               12

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv_udp";

typedef struct {
    uint16_t len;
    char data[LINE_MAX_LEN];
} csi_line_t;

static QueueHandle_t s_queue;
static esp_netif_t *s_sta_netif;
static volatile bool s_connected = false;

/* compteurs pour les statistiques affichees sur le port serie */
static volatile uint32_t s_csi_count = 0, s_queue_full = 0, s_sent = 0, s_send_err = 0;
/* diagnostic : toutes les trames CSI recues (avant le filtre MAC) et la derniere vue */
static volatile uint32_t s_all_count = 0;
static volatile int s_last_cwb = -1, s_last_sec = -1;
static uint8_t s_last_mac[6];

/* ---------- mise en forme rapide de la ligne (sans snprintf par valeur) ---------- */
static int put_int(char *p, int v)
{
    char tmp[12];
    int n = 0, len = 0;
    unsigned int u = (v < 0) ? (unsigned int)(-v) : (unsigned int)v;
    if (v < 0) {
        p[len++] = '-';
    }
    do {
        tmp[n++] = '0' + (u % 10);
        u /= 10;
    } while (u);
    while (n) {
        p[len++] = tmp[--n];
    }
    return len;
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        return;
    }
    s_all_count++;
    s_last_cwb = info->rx_ctrl.cwb;
    s_last_sec = info->rx_ctrl.secondary_channel;
    memcpy(s_last_mac, info->mac, 6);
    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;                                 /* on ne garde que les trames de l'emetteur S */
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (s_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
#endif
    s_count++;
    s_csi_count++;

    if (!s_connected) {
        return;
    }

    static csi_line_t line;                     /* callback non reentrant (tache Wi-Fi) */
    uint32_t rx_id = *(uint32_t *)(info->payload + 15);

    /* meme format que csi_recv (ESP32-S3) : 24 champs, puis la liste des valeurs */
    int n = snprintf(line.data, LINE_MAX_LEN,
                     "CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,\"[",
                     (int)rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
                     rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
                     rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
                     rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
                     (int)rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->sig_mode,
                     info->len, info->first_word_invalid);
    if (n <= 0 || n >= LINE_MAX_LEN) {
        return;
    }
    for (int i = 0; i < info->len && n < LINE_MAX_LEN - 16; i++) {
        if (i) {
            line.data[n++] = ',';
        }
        n += put_int(line.data + n, (int16_t)(compensate_gain * info->buf[i]));
    }
    line.data[n++] = ']';
    line.data[n++] = '"';
    line.data[n++] = '\n';
    line.len = n;

    if (xQueueSend(s_queue, &line, 0) != pdTRUE) {
        s_queue_full++;                         /* le reseau ne suit pas : on perd la trame */
    }
}

/* ---------- tache d'envoi UDP ---------- */
static void udp_task(void *arg)
{
    static csi_line_t item;
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() a echoue : errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
    struct sockaddr_in local = {
        .sin_family = AF_INET,
        .sin_port = htons(UDP_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    bind(sock, (struct sockaddr *)&local, sizeof(local));

    struct sockaddr_in pc = {0};
    bool pc_known = false;
    char hello[32];
    TickType_t last_stats = xTaskGetTickCount();
    uint32_t last_count = 0, last_sent = 0, last_all = 0;

    while (1) {
        /* 1) le PC s'est-il annonce ? (lecture non bloquante) */
        struct sockaddr_in from;
        socklen_t fromlen = sizeof(from);
        int r = recvfrom(sock, hello, sizeof(hello) - 1, MSG_DONTWAIT, (struct sockaddr *)&from, &fromlen);
        if (r > 0) {
            hello[r] = 0;
            if (strncmp(hello, "WISEE_HELLO", 11) == 0) {
                if (!pc_known || pc.sin_addr.s_addr != from.sin_addr.s_addr) {
                    ESP_LOGI(TAG, "PC trouve : %s:%d", inet_ntoa(from.sin_addr), ntohs(from.sin_port));
                }
                pc = from;
                pc_known = true;
            }
        }

        /* 2) envoi des lignes CSI en attente */
        if (xQueueReceive(s_queue, &item, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (pc_known && s_connected) {
                if (sendto(sock, item.data, item.len, 0, (struct sockaddr *)&pc, sizeof(pc)) < 0) {
                    s_send_err++;
                    if (errno == ENOMEM) {
                        vTaskDelay(1);          /* tampons lwIP pleins : on laisse respirer */
                    }
                } else {
                    s_sent++;
                }
            }
        }

        /* 3) statistiques toutes les 5 s sur le port serie */
        if (xTaskGetTickCount() - last_stats >= pdMS_TO_TICKS(5000)) {
            float dt = (xTaskGetTickCount() - last_stats) * portTICK_PERIOD_MS / 1000.0f;
            ESP_LOGI(TAG, "RX%d | Wi-Fi %s | PC %s | CSI %.1f/s | envoyees %.1f/s | file pleine %lu | erreurs envoi %lu",
                     RX_ID, s_connected ? "OK" : "deconnecte", pc_known ? inet_ntoa(pc.sin_addr) : "inconnu (pas de HELLO)",
                     (s_csi_count - last_count) / dt, (s_sent - last_sent) / dt,
                     (unsigned long)s_queue_full, (unsigned long)s_send_err);
            uint8_t ch = 0;
            wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
            wifi_bandwidth_t bw = WIFI_BW_HT20;
            esp_wifi_get_channel(&ch, &sec);
            esp_wifi_get_bandwidth(WIFI_IF_STA, &bw);
            ESP_LOGI(TAG, "   diag : canal %d, secondaire %s, %s | toutes trames CSI %.1f/s, derniere de " MACSTR " (cwb %d, sec %d)",
                     ch, sec == WIFI_SECOND_CHAN_BELOW ? "dessous" : (sec == WIFI_SECOND_CHAN_ABOVE ? "dessus" : "aucun"),
                     bw == WIFI_BW_HT40 ? "40 MHz" : "20 MHz", (s_all_count - last_all) / dt,
                     MAC2STR(s_last_mac), s_last_cwb, s_last_sec);
            last_stats = xTaskGetTickCount();
            last_count = s_csi_count;
            last_sent = s_sent;
            last_all = s_all_count;
        }
    }
}

/* ---------- Wi-Fi station + IP fixe ---------- */
static void event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        esp_netif_dhcpc_stop(s_sta_netif);
        esp_netif_ip_info_t ip = {0};
        ip.ip.addr = ESP_IP4TOADDR(192, 168, 4, 200 + RX_ID);
        ip.netmask.addr = ESP_IP4TOADDR(255, 255, 255, 0);
        ip.gw.addr = ESP_IP4TOADDR(192, 168, 4, 1);
        if (esp_netif_set_ip_info(s_sta_netif, &ip) == ESP_OK) {
            s_connected = true;
            ESP_LOGI(TAG, "Connecte a %s, IP fixe 192.168.4.%d", WISEE_SSID, 200 + RX_ID);
        }
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected = false;
        ESP_LOGW(TAG, "Deconnecte du point d'acces, nouvelle tentative...");
        esp_wifi_connect();
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    s_sta_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

    wifi_config_t sta = {0};
    strcpy((char *)sta.sta.ssid, WISEE_SSID);
    strcpy((char *)sta.sta.password, WISEE_PASSWORD);
    sta.sta.channel = WISEE_CHANNEL;
    sta.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));

    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));   /* pas de veille : sinon on rate des trames */

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "RX%d, MAC " MACSTR, RX_ID, MAC2STR(mac));
}

static void wifi_esp_now_init(void)
{
    /* repris de csi_recv ; non bloquant si l'ajout du pair echoue */
    esp_now_peer_info_t peer = {
        .channel   = 0,                         /* 0 = canal courant (celui du point d'acces) */
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };
    ESP_ERROR_CHECK_WITHOUT_ABORT(esp_now_init());
    ESP_ERROR_CHECK_WITHOUT_ABORT(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    ESP_ERROR_CHECK_WITHOUT_ABORT(esp_now_add_peer(&peer));
}

static void wifi_csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_queue = xQueueCreate(QUEUE_LEN, sizeof(csi_line_t));
    assert(s_queue);

    wifi_init();
    wifi_esp_now_init();
    wifi_csi_init();

    xTaskCreate(udp_task, "udp_task", 4096, NULL, 5, NULL);
}
