# Datasets du projet

Ce document décrit les jeux de données présents dans le dossier `data/` et leur rôle dans le projet de prétraitement CSI ESP32.

---

## 1. Vue d’ensemble

Le projet contient plusieurs types de données liées à la mesure Wi‑Fi CSI (Channel State Information) sur ESP32 :

- données d’activité humaine
- données de respiration
- données de transmission / réception dans des scénarios de test
- fichiers d’exemple pour le prétraitement

La plupart des fichiers sont de type `.txt` et suivent la structure CSI brute :

```text
timestamp_ms, v0, v1, ..., v127
```

Chaque ligne représente un paquet capturé. Les valeurs sont utilisées pour reconstruire les sous-porteuses CSI et calculer l’amplitude / la phase.

---

## 2. Arborescence générale

```text
data/
├── active_ap/
├── active_sta/
├── Datasets/
├── passive/
└── ...
```

Chaque dossier correspond à un type d’expérience ou à un mode de capture.

---

## 3. Dossier `data/Datasets/`

C’est le dossier principal pour les données d’exploitation de la recherche.

### Structure typique

```text
data/Datasets/
├── README.md
├── x_trn.npy
├── y_trn.npy
├── Activity/
│   ├── FA/
│   ├── GU/
│   ├── LD/
│   ├── SD/
│   └── WA/
├── Breathing(experimental)/
│   └── Breathing_Rate_Monitoring_Dataset/
│       ├── 10BPM/
│       ├── 15BPM/
│       └── 20BPM/
```

### Contenu

#### `x_trn.npy` et `y_trn.npy`

Ce sont probablement des fichiers NumPy déjà préparés pour l’entraînement d’un modèle.

- `x_trn.npy` : features d’entrée (ex. signaux CSI transformés)
- `y_trn.npy` : labels correspondants (ex. activité ou classe)

Ils servent à alimenter un entraînement ML ou une classification.

#### `Activity/`

Dossier contenant des données d’activité humaine classées par type d’activité :

- `FA` : Fall / Chute
- `GU` : ??? (selon le dataset, activité de type “gait”, “guard”, etc. — à confirmer selon le protocole de collecte)
- `LD` : Lying Down / position couchée
- `SD` : Sitting Down / assis
- `WA` : Walking / marche

Chaque activité contient des sous-dossiers `s1`, `s2`, `s3`, ..., qui représentent probablement des sujets ou des sessions d’acquisition.

Exemple :

```text
Activity/FA/s1/
```

Ce dossier contient plusieurs fichiers texte de captures CSI pour le sujet/session donné.

---

## 4. Dossier `Activity/`

C’est le jeu de données le plus important pour l’analyse d’activité.

### Exemple de structure

```text
Activity/
├── FA/
│   ├── s1/
│   │   ├── csi_data_1720464506.txt
│   │   ├── csi_data_1720464592.txt
│   │   └── ...
│   ├── s2/
│   └── ...
├── GU/
├── LD/
├── SD/
└── WA/
```

### Sens de chaque activité

- `FA` : fall / chute
- `GU` : autre activité humaine ou posture
- `LD` : lying down / allongé
- `SD` : sitting down / assis
- `WA` : walking / marche

Il existe généralement un fichier par capture temporelle. Chaque fichier est une séquence CSI brute pour une actions ou une période de mesure.

---

## 5. Dossier `Breathing(experimental)/`

Ce dossier contient un dataset expérimental pour la respiration.

```text
Breathing(experimental)/
└── Breathing_Rate_Monitoring_Dataset/
    ├── 10BPM/
    ├── 15BPM/
    └── 20BPM/
```

### Signification

Les sous-dossiers `10BPM`, `15BPM`, `20BPM` indiquent probablement :

- 10 battements par minute
- 15 battements par minute
- 20 battements par minute

Cela correspond à des données de respiration synthétiques ou contrôlées, utiles pour valider la détection de fréquence respiratoire.

Ce type de dataset est particulièrement adapté à la configuration :

```bash
python csi_preprocess.py data/Datasets/Breathing* -o out_breath/ --mode breathing -j 8
```

---

## 6. Dossier `active_ap/`

Le dossier `active_ap/` est probablement un exemple de capture où un point d’accès est actif et où un signal CSI est mesuré dans un contexte d’activité.

Structure vue dans le projet :

```text
active_ap/
├── CMakeLists.txt
├── README.md
└── main/
    ├── CMakeLists.txt
    ├── component.mk
    ├── Kconfig.projbuild
    └── main.cc
```

### Rôle

Ce dossier est plus orienté développement / firmware que dataset d’analyse. Il contient probablement le code ESP32 utilisé pour générer ou acquérir les traces CSI sur un point d’accès actif.

---

## 7. Dossier `active_sta/`

Même logique que `active_ap/`, mais dans un contexte de station client active.

Structure :

```text
active_sta/
├── CMakeLists.txt
├── README.md
└── main/
    ├── CMakeLists.txt
    ├── component.mk
    ├── Kconfig.projbuild
    └── main.cc
```

### Rôle

Il permet probablement de configurer un ESP32 en mode client actif pour la collecte CSI.

---

## 8. Dossier `passive/`

Le dossier `passive/` est probablement une configuration passive de collecte CSI.

```text
passive/
├── CMakeLists.txt
├── README.md
└── main/
    ├── CMakeLists.txt
    ├── component.mk
    ├── Kconfig.projbuild
    └── main.cc
```

### Rôle

Il correspond à un mode de capture où l’ESP32 écoute passivement le trafic Wi‑Fi sans participer activement à la communication. Cela peut être utile pour des études de sensing sans accès direct à la station ou au point d’accès.

---

## 9. Fichiers `.txt` de capture

Les fichiers `csi_data_*.txt` sont les données brutes utilisées par le script de prétraitement.

### Exemple

```text
csi_data_1720464506.txt
```

### Leur format

Chaque fichier contient une série de lignes du type :

```text
1720464506, 0.123, 0.456, ...
```

Le script `csi_preprocess.py` lit ces lignes, les nettoie, puis produit des séries temporelles exploitable pour :

- la détection d’activité
- la respiration
- l’analyse spectrale
- la classification

---

## 10. Comment les utiliser dans le projet

### Exemple 1 : activité

```bash
python csi_preprocess.py data/Datasets/Activity -o out_activity --mode activity --plot
```

### Exemple 2 : respiration

```bash
python csi_preprocess.py data/Datasets/Breathing* -o out_breath --mode breathing -j 8
```

### Exemple 3 : fichier unique

```bash
python csi_preprocess.py data/Datasets/Activity/FA/s1/csi_data_1720464506.txt -o out_single --mode activity --plot
```

---

## 11. Ce que représente chaque dataset

| Dossier | Type | Utilisation probable |
|---|---|---|
| `data/Datasets/Activity/` | Activité humaine | classification des gestes ou mouvements |
| `data/Datasets/Breathing(experimental)/` | Respiration | estimation de fréquence respiratoire |
| `active_ap/` | firmware ESP32 | acquisition Wi‑Fi en mode AP |
| `active_sta/` | firmware ESP32 | acquisition Wi‑Fi en mode station |
| `passive/` | firmware ESP32 | collecte passive |
| `x_trn.npy`, `y_trn.npy` | données ML | entraînement de modèle |

---

## 12. Interprétation pratique

En pratique, ces datasets servent à tester le pipeline de prétraitement avant d’entrer dans une phase de :

- extraction de features
- segmentation temporelle
- apprentissage supervisé
- classification d’activité
- estimation de respiration

Le script traite les fichiers textuels de manière robuste et produit des données standardisées prêtes à l’analyse.

---

## 13. Résumé

Le projet contient à la fois :

- des données brutes CSI pour l’analyse
- des données préparées pour l’apprentissage
- des exemples de firmware ESP32 pour acquérir de nouvelles mesures

Les jeux de données les plus utiles pour le prétraitement sont surtout :

- `data/Datasets/Activity/`
- `data/Datasets/Breathing(experimental)/`
- et les fichiers `csi_data_*.txt` qu’ils contiennent

---
