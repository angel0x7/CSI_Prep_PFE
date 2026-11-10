# Acquisition CSI WISEE (partie systèmes embarqués)

Ce dossier contient la partie **acquisition** de WISEE : les firmwares ESP32-S3, les scripts
d'enregistrement et nos sessions déjà exportées en CSV, au même format que `data/Datasets/`.
Ces CSV passent donc directement dans `csi_preprocess.py`.

```
acquisition/
├── firmware/            ESP32-S3 (ESP-IDF 5.5.5)
│   ├── README_UDP.md    procédure complète : flasher, connecter le PC, enregistrer
│   ├── wisee_ap/        point d'accès Wi-Fi WISEE_CSI (canal 11, 40 MHz)
│   └── csi_recv_udp/    récepteurs RX1..RX4 : CSI de l'émetteur -> UDP vers le PC
├── python/              enregistrement et export
│   ├── record_multi.py  enregistre N récepteurs (--udp ou --serial) et exporte les CSV
│   ├── export_csv.py    convertit des sessions brutes (.txt) en CSV au format de l'équipe IA
│   ├── enregistrer_dataset.ps1, indexer_dataset.py   prises du dataset en boucle + index.csv
│   ├── record_csi.py, analyze_session.py, compare_sessions.py, simulate_esp32.py
│   ├── wisee_csi/       parser des lignes CSI_DATA, métriques, modèle de canal du simulateur
│   └── tests/
└── data/                nos sessions (CSV uniquement)
```

## Le système

- **Émetteur S** (`esp-csi/examples/get-started/csi_send`, non modifié) : il envoie environ 90 trames/s en ESP-NOW, canal 11, 40 MHz.
- **4 récepteurs** (`csi_recv_udp`) dans les coins de la pièce : chacun mesure le CSI des trames de S et l'envoie au PC en UDP, via le point d'accès `wisee_ap`.
- **PC** :
  ```
  python record_multi.py --udp 5005 --map 192.168.4.201=RX1 192.168.4.202=RX2 192.168.4.203=RX3 192.168.4.204=RX4 \
         --duration 20 --outdir ../data/AAAA-MM-JJ_campagne --name WA_sujet_01 --note "marche, sujet X, essai 1"
  ```

Chaque session crée un dossier contenant :

| Fichier | Contenu |
|---|---|
| `RXn.csv` | **Format de l'équipe IA** (voir plus bas) |
| `RXn.txt` | Brut complet : heure PC + ligne `CSI_DATA` (384 valeurs : LLTF 64 + HT-LTF 128 sous-porteuses), RSSI, numéro de trame de S. Non versionné (trop lourd) |
| `summary.json` | Date, note, trames/s, pertes par récepteur, part de trames vues par tous les récepteurs |

## Format des CSV

Une ligne par trame, **129 colonnes, sans en-tête** (même format que `data/Datasets/Activity/…`) :

```
timestamp_ms, v0, v1, ..., v127
```

- **`timestamp_ms`** : heure Unix en ms, tirée de l'horloge de l'ESP32 et calée sur l'heure du PC à la première trame. Elle est régulière, environ **90 Hz** (dt ≈ 11 ms). `csi_preprocess.py --fs` rééchantillonne.
- **`v0..v127`** : les **64 sous-porteuses du LLTF**, pour chacune la partie imaginaire puis la partie réelle, dans l'**ordre ESP32 standard 0..31, −32..−1**.
  - Les sous-porteuses nulles sont aux indices 27 à 37, comme dans `data/Datasets`.
  - L'indice 0 (la fréquence centrale) vaut 0 chez nous.
  - En 40 MHz, l'ESP32-S3 fournit le LLTF dans l'ordre −32..31 : `export_csv.py` remet les deux moitiés dans l'ordre standard.
- **Un fichier par récepteur.** Pour aligner les 4 récepteurs trame par trame, utiliser le numéro de trame de S, présent dans les `.txt`.

## Sessions disponibles (`data/`)

| Campagne | Matériel | Sessions |
|---|---|---|
| `2026-10-14_1RX` | 1 récepteur, liaison série | essais techniques, puis vide / marche / assis, avec gain libre (P1–P3) et gain figé (T2.1–T2.3) |
| `2026-10-14_2RX` | 2 récepteurs, S au centre | L1 vide, L2 marche entre S et RX1, L3 marche entre S et RX2, L1bis vide |
| `2026-10-21_soutenance_inter` | 2 récepteurs | D1 personne immobile, D2 allers-retours (60 s), passage devant le jury + répétition |
| `2026-11-04_3RX_UDP` | **3 récepteurs sans fil (UDP)**, S et point d'accès au centre, RX1–RX3 éloignés dans la pièce | `test_transport_30s`, `vide` (début perturbé), `vide_sortie_15s` (pièce vraiment vide : référence), `marche_allers-retours` (60 s chacune) |
| **`DATASET`** | **4 récepteurs en UDP** | **campagne du 10/11 : 294 prises de 8 s rangées en `<activité>/p<nombre de personnes>/prise_NNN` (chute, assis, debout, marche, vide ; 1 à 5 personnes). Pertes Wi-Fi élevées (11-96 %) : lire `data/DATASET/README.md` et la colonne `qualite` de `index.csv` avant usage.** |

Pour les prochaines campagnes, on garde la convention du `DATASET` : `<activité>/p<nombre de personnes>/prise_NNN`, prises enregistrées en boucle avec `python/enregistrer_dataset.ps1`, puis `python/indexer_dataset.py` pour regénérer `index.csv`.

## Installation Python

```
cd acquisition/python
python -m venv venv
venv\Scripts\activate          (Windows)
pip install numpy matplotlib pyserial
python -m unittest discover tests
```
