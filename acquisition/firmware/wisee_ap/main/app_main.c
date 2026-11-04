/*
 * WISEE - point d'acces Wi-Fi (carte de secours ESP32-S3)
 *
 * Reseau local du systeme : SSID WISEE_CSI, canal 11, 40 MHz (HT40, canal secondaire en dessous),
 * soit exactement le canal de l'emetteur csi_send. Les recepteurs (csi_recv_udp) s'y connectent
 * avec une IP fixe 192.168.4.201 a .204, le PC s'y connecte en DHCP (192.168.4.2 a .101).
 * Le point d'acces est 192.168.4.1.
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"

#define WISEE_SSID       "WISEE_CSI"
#define WISEE_PASSWORD   "wisee2026"            /* 8 caracteres minimum (WPA2) */
#define WISEE_CHANNEL    11
#define MAX_STATIONS     8                      /* 4 recepteurs + PC + marge */

static const char *TAG = "wisee_ap";

static void event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *e = (wifi_event_ap_staconnected_t *)data;
        ESP_LOGI(TAG, "Appareil connecte : " MACSTR " (AID %d)", MAC2STR(e->mac), e->aid);
    } else if (id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *e = (wifi_event_ap_stadisconnected_t *)data;
        ESP_LOGW(TAG, "Appareil deconnecte : " MACSTR " (AID %d)", MAC2STR(e->mac), e->aid);
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));

    wifi_config_t ap = {0};
    strcpy((char *)ap.ap.ssid, WISEE_SSID);
    ap.ap.ssid_len = strlen(WISEE_SSID);
    strcpy((char *)ap.ap.password, WISEE_PASSWORD);
    ap.ap.channel = WISEE_CHANNEL;
    ap.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ap.ap.max_connection = MAX_STATIONS;
    ap.ap.pmf_cfg.required = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT40));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    /* canal 11 en 40 MHz : le canal secondaire ne peut etre qu'en dessous (11-) */
    esp_err_t ch_err = esp_wifi_set_channel(WISEE_CHANNEL, WIFI_SECOND_CHAN_BELOW);
    ESP_LOGI(TAG, "esp_wifi_set_channel(11, dessous) -> %s", esp_err_to_name(ch_err));

    uint8_t primary;
    wifi_second_chan_t second;
    wifi_bandwidth_t bw;
    esp_wifi_get_channel(&primary, &second);
    esp_wifi_get_bandwidth(WIFI_IF_AP, &bw);
    ESP_LOGI(TAG, "Point d'acces %s pret : canal %d, secondaire %s, %s, IP 192.168.4.1",
             WISEE_SSID, primary,
             second == WIFI_SECOND_CHAN_BELOW ? "en dessous" : (second == WIFI_SECOND_CHAN_ABOVE ? "au dessus" : "aucun"),
             bw == WIFI_BW_HT40 ? "40 MHz" : "20 MHz");

    while (1) {
        wifi_sta_list_t list;
        esp_wifi_get_channel(&primary, &second);
        esp_wifi_get_bandwidth(WIFI_IF_AP, &bw);
        if (esp_wifi_ap_get_sta_list(&list) == ESP_OK) {
            ESP_LOGI(TAG, "%d appareil(s) connecte(s) | canal %d, secondaire %s, %s", list.num, primary,
                     second == WIFI_SECOND_CHAN_BELOW ? "en dessous" : (second == WIFI_SECOND_CHAN_ABOVE ? "au dessus" : "aucun"),
                     bw == WIFI_BW_HT40 ? "40 MHz" : "20 MHz");
        }
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
