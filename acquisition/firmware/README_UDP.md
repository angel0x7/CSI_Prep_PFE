# WISEE – passage au sans-fil (UDP)

## Principe

```
            ESP-NOW canal 11 HT40 (inchangé)
   S (csi_send) ─────────────────────────────►  RX1..RX4 (csi_recv_udp)
                                                  │  Wi-Fi station, IP fixe 192.168.4.201..204
                                                  │  1 ligne CSI_DATA = 1 paquet UDP
                                                  ▼
                              AP (wisee_ap, carte de secours)  SSID WISEE_CSI, canal 11 HT40, 192.168.4.1
                                                  │
                                                  ▼
                              PC (Wi-Fi WISEE_CSI, DHCP)  record_multi.py --udp 5005
```

- **L'émetteur S ne change pas** : on garde csi_send tel quel.
- **Le point d'accès (AP)** est sur le **même canal que S** (11, 40 MHz, canal secondaire en dessous). C'est obligatoire : un récepteur connecté à un AP écoute le canal de cet AP.
- **Les récepteurs** captent le CSI exactement comme avant et l'envoient en UDP au PC. Les lignes ont le même format qu'avant, donc tous nos scripts Python marchent sans modification.
  - Le PC se fait connaître en envoyant `WISEE_HELLO` à chaque récepteur, une fois par seconde. `record_multi.py --udp` le fait tout seul. Tant qu'un récepteur n'a pas reçu de HELLO, il n'envoie rien.
- **Chaque récepteur garde sa MAC d'usine.** L'ancien csi_recv forçait la MAC 1a:00:…, ce qui est impossible à 4 sur le même AP.
- **Une seule différence entre les 4 récepteurs : `RX_ID`** (ligne `#define RX_ID` dans `csi_recv_udp/main/app_main.c`). Il fixe l'IP de la carte : RX1 = 192.168.4.201, …, RX4 = 192.168.4.204.

## 0. Ouvrir l'environnement ESP-IDF (PowerShell)

```
. C:\Espressif\tools\Microsoft.v5.5.5.PowerShell_profile.ps1
cd <depot>\acquisition\firmware
```

## 1. Point d'accès : carte de secours (une seule fois)

```
cd wisee_ap
idf.py set-target esp32s3
idf.py -p COMx flash monitor
```

Dans le monitor, on doit voir :

`Point d'acces WISEE_CSI pret : canal 11, secondaire en dessous, 40 MHz, IP 192.168.4.1`

Ensuite, toutes les 10 s, le nombre d'appareils connectés s'affiche. Quitter le monitor : Ctrl+T puis X.

La carte n'a besoin que d'être alimentée (USB du PC ou chargeur). Antenne obligatoire.

## 2. Récepteurs : une carte à la fois

```
cd ..\csi_recv_udp
idf.py set-target esp32s3          # seulement la toute première fois
```

Pour **chaque** carte :

1. Dans `main\app_main.c`, mettre `#define RX_ID 1` (puis 2, 3, 4 pour les cartes suivantes).
2. Flasher :
   ```
   idf.py -p COMx flash monitor
   ```
   La compilation ne prend que quelques secondes après la première.
3. Vérifier dans le monitor :
   - `RX1, MAC xx:xx:…` puis `Connecte a WISEE_CSI, IP fixe 192.168.4.201` ;
   - toutes les 5 s : `RX1 | Wi-Fi OK | PC inconnu (pas de HELLO) | CSI 90.0/s | envoyees 0.0/s …`.

   `CSI ≈ 90/s` prouve que la carte capte bien S, même en étant connectée à l'AP.
4. **Coller une étiquette RX1… sur la carte.**

Si `CSI 0.0/s` : S n'est pas allumé, ou l'AP n'est pas sur le canal 11 en 40 MHz (voir le message de l'AP).

## 3. PC

1. Se connecter au Wi-Fi **WISEE_CSI** (mot de passe `wisee2026`). Il n'y a pas d'internet sur ce réseau : utiliser l'Ethernet ou le téléphone pour le reste.
2. `ipconfig` : la carte Wi-Fi doit avoir une adresse 192.168.4.x (souvent 192.168.4.2).
3. Autoriser l'UDP entrant, **une seule fois**, dans un PowerShell **administrateur** :
   ```
   netsh advfirewall firewall add rule name="WISEE UDP 5005" dir=in action=allow protocol=UDP localport=5005
   ```
4. Test : `ping 192.168.4.201` doit répondre.

## 4. Enregistrer

```
cd <depot>\acquisition\python
venv\Scripts\activate
python record_multi.py --udp 5005 --map 192.168.4.201=RX1 --duration 30 --name udp_test1
```

Avec les 4 récepteurs :

```
python record_multi.py --udp 5005 --map 192.168.4.201=RX1 192.168.4.202=RX2 192.168.4.203=RX3 192.168.4.204=RX4 --duration 60 --name udp_4rx
```

Le monitor de chaque récepteur doit alors passer à `PC 192.168.4.x | … envoyees ~90/s`.

## 5. Critères de validation

On compare avec les sessions série du 14 et du 21/10 :

| | Série (référence) | Objectif UDP |
|---|---|---|
| Trames/s par RX | ~90 | ~90 |
| Pertes par RX | 0,1–2 % | < 3 % |
| Trames vues par tous les RX | 97–99 % | > 95 % |

Ensuite, on débranche les RX du PC et on les alimente par des chargeurs ou des batteries, dans les coins de la pièce.

## Dépannage

| Symptôme (monitor du RX) | Cause probable |
|---|---|
| `Deconnecte du point d'acces` en boucle | AP éteint, mauvais mot de passe, ou carte trop loin de l'AP |
| `PC inconnu (pas de HELLO)` alors que record_multi tourne | Pare-feu Windows, PC pas sur WISEE_CSI, ou mauvaise IP dans `--map` |
| `file pleine` qui augmente | Le Wi-Fi ne suit pas : noter les chiffres, on passera à un format binaire plus compact |
| `erreurs envoi` qui augmente | Idem, ou PC parti du réseau |
| `CSI 0.0/s` | S éteint, ou canal/bande de l'AP différents de S |

## Fichiers

- `wisee_ap/` : firmware du point d'accès.
- `csi_recv_udp/` : firmware des récepteurs (base csi_recv d'Espressif, modifiée).
- `../python/record_multi.py` : envoie `WISEE_HELLO` en mode `--udp` et exporte les CSV à la fin de chaque session.
- L'ancien firmware série est `esp-csi/examples/get-started/csi_recv` (Espressif) : pour revenir en arrière, il suffit de le reflasher.
