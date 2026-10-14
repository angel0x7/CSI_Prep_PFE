# PFE_CSI_Cleaning

Ce projet contient un pipeline Python pour nettoyer, filtrer et exporter des données CSI ESP32 avant de les utiliser dans des tâches de détection d’activité, de respiration, de chute ou d’analyse de mouvement.

Le script principal est :

- [csi_preprocess.py](csi_preprocess.py)

Il lit des fichiers CSI bruts, supprime les artefacts, applique des filtres, rééchantillonne les signaux et exporte les données nettoyées en format `.npz` ou `.csv`.

---

## 1. Objectif

Le code prend en entrée des traces de CSI au format texte brut, généralement sous la forme :

```text
timestamp_ms, v0, v1, ..., v127
```

Chaque ligne correspond à un paquet Wi‑Fi capture, et les valeurs sont organisées comme suit :

- 1 colonne : horodatage en millisecondes
- 128 colonnes restantes : 64 sous-porteuses CSI, chaque sous-porteuse étant représentée par une partie imaginaire et une partie réelle

Le script réalise ensuite :

1. lecture robuste des fichiers
2. rejet des lignes corrompues ou incohérentes
3. nettoyage des sous-porteuses nulles ou anormales
4. calcul de l’amplitude et éventuellement de la phase
5. suppression des valeurs aberrantes (Hampel)
6. rééchantillonnage temporel uniforme
7. filtrage Butterworth
8. normalisation optionnelle
9. export du résultat

---

## 2. Prérequis

Le projet dépend de :

- Python 3.x
- NumPy
- SciPy
- matplotlib (optionnel, uniquement si `--plot` est activé)

Installation recommandée :

```bash
pip install numpy scipy matplotlib
```

---

## 3. Structure du projet

```text
PFE_CSI_Cleaning/
├── csi_preprocess.py
├── README.md
├── data/
│   ├── Datasets/
│   ├── active_ap/
│   ├── active_sta/
│   ├── passive/
│   └── ...
└── output/
```

Le dossier `data/Datasets` contient les exemples de fichiers CSI utilisés pour les essais.

---

## 4. Utilisation du script

### 4.1 Traitement d’un seul fichier

```bash
python csi_preprocess.py chemin/vers/fichier.txt -o sortie/
```

Exemple :

```bash
python csi_preprocess.py data/Datasets/Activity/FA/s1/csi_data_1720464506.txt -o out/
```

### 4.2 Traitement d’un dossier entier

```bash
python csi_preprocess.py data/Datasets/Activity -o out/ --mode activity -j 8
```

Le script parcourt récursivement les fichiers du dossier selon le motif défini par défaut :

```text
csi_data_*.txt
```

Tu peux changer ce motif avec :

```bash
--pattern "*.txt"
```

---

## 5. Modes prédéfinis

Le script propose deux presets de filtrage utiles :

### `activity`

```bash
--mode activity
```

Paramètres par défaut :

- fréquence basse : 0.3 Hz
- fréquence haute : 10 Hz

Conçu pour des mouvements de corps ou d’activité globale (marche, chute, assis, etc.).

### `breathing`

```bash
--mode breathing
```

Paramètres par défaut :

- fréquence basse : 0.1 Hz
- fréquence haute : 0.6 Hz

Conçu pour la respiration, car les fréquences respiratoires sont généralement autour de 0.1 à 0.6 Hz.

---

## 6. Paramètres utiles

### Fréquence d’échantillonnage

```bash
--fs 50
```

La valeur par défaut est `50 Hz`.

### Bande passante personnalisée

```bash
python csi_preprocess.py fichier.txt -o out/ --low 0.5 --high 8
```

### Ordre du filtre Butterworth

```bash
--order 4
```

### Sous-porteuses conservées

```bash
--keep-sc "2-26,42-63"
```

Par défaut, le script ne garde que les sous-porteuses fiables, en éliminant les zones inutiles (DC, garde band, etc.).

### Désactiver Hampel

```bash
--no-hampel
```

### Désactiver le contrôle des lignes d’en-tête

```bash
--no-header-check
```

### Normalisation

```bash
--normalize zscore
```

Options disponibles :

- `none`
- `zscore`
- `minmax`

### Inclure la phase

```bash
--phase
```

Le script calcule aussi une version filtrée de la phase, si besoin.

### Générer les figures de contrôle

```bash
--plot
```

Cela produit un PNG pour chaque fichier traité, avec :

- signal brut
- signal nettoyé
- données filtrées
- spectre de puissance

---

## 7. Exemples complets

### Exemple 1 : traitement simple d’un fichier

```bash
python csi_preprocess.py data/Datasets/Activity/FA/s1/csi_data_1720464506.txt -o out_activity --mode activity --plot
```

### Exemple 2 : respiration

```bash
python csi_preprocess.py data/Datasets/Breathing\(experimental\)/Breathing_Rate_Monitoring_Dataset -o out_breath --mode breathing -j 8
```

### Exemple 3 : personnalisation avancée

```bash
python csi_preprocess.py data/Datasets/Activity -o out_custom --low 0.5 --high 8 --normalize zscore --phase --format csv
```

### Exemple 4 : traitement parallèle sur plusieurs fichiers

```bash
python csi_preprocess.py data/Datasets/Activity -o out_multi --mode activity -j 4
```

---

## 8. Sorties produites

Le script crée un dossier de sortie. Selon les paramètres, il peut générer :

- `summary.csv` : résumé global de tous les fichiers traités
- fichiers `.npz` : données nettoyées et filtrées
- fichiers `.csv` : version tabulaire exportée
- fichiers `.png` : graphiques de contrôle si `--plot` est activé

### Exemple de structure de sortie

```text
out/
├── summary.csv
├── csi_data_1720464506.npz
├── csi_data_1720464506.png
└── ...
```

---

## 9. Ce que contiennent les données exportées

Dans les fichiers `.npz`, tu peux retrouver des arrays comme :

- `t` : temps en secondes
- `amp` : amplitude CSI filtrée
- `amp_clean` : amplitude après nettoyage et rééchantillonnage
- `subcarriers` : indices des sous-porteuses conservées
- `subcarrier_freq_idx` : indices fréquentiels signés
- `fs` : fréquence d’échantillonnage
- `phase` : phase filtrée (si `--phase` est utilisé)

Cela permet ensuite de construire un dataset exploitable par des modèles ML ou des analyses temporelles.

---

## 10. Pipeline interne en résumé

Le code applique successivement :

1. lecture des fichiers et gestion des erreurs
2. filtrage des lignes incohérentes
3. reconstruction du CSI complexe
4. calcul de l’amplitude
5. détection des valeurs aberrantes avec Hampel
6. interpolation sur une grille temporelle régulière
7. application d’un filtre Butterworth
8. normalisation optionnelle
9. export pour analyse ou apprentissage

---

## 11. Bonnes pratiques

- Utilise `--mode activity` pour des mouvements globaux
- Utilise `--mode breathing` pour des signaux lents
- Active `--plot` au début pour vérifier visuellement le bon traitement
- Vérifie `summary.csv` pour détecter les fichiers problématiques
- Si les données sont très bruitées, essaie d’ajuster `--hampel-sigma` ou `--keep-sc`

---

## 12. Dépannage rapide

### Le script ne traite rien

Vérifie que les chemins sont corrects et que les fichiers correspondent au motif attendu.

### Erreur sur le filtre

Cela peut venir d’un signal trop court ou d’une bande passante invalide.

Vérifie notamment :

- `--low < --high`
- `--high < fs/2`
- durée suffisante du signal

### Données trop bruitées

Essaie :

```bash
--no-hampel
```

ou ajuste :

```bash
--hampel-sigma 3
```

---

## 13. Commande de démarrage rapide

Si tu veux lancer directement le traitement sur un dossier de données de test :

```bash
python csi_preprocess.py data/Datasets/Activity -o output_activity --mode activity --plot --normalize zscore
```

C’est une commande simple, robuste et adaptée pour un premier test.

---

## 14. Conclusion

Ce script est un outil de prétraitement pour données CSI ESP32. Il permet de convertir des traces brutes en données utilisables pour une analyse de mouvement, une détection d’activité ou une estimation respiratoire.

Le point clé est que le prétraitement est important : sans nettoyage et filtrage, les signaux CSI sont trop bruités pour être exploités de manière fiable.

---

Pour aller plus loin, tu peux ensuite :

- extraire des features statistiques
- créer un dataset pour un modèle ML
- faire de la classification d’activité
- estimer le rythme respiratoire

