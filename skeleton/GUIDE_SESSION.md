# Tester le modèle et enregistrer des données dans un nouveau lieu

Une seule commande fait les deux : elle **teste le modèle en direct** et **enregistre des données étiquetées**
qui serviront ensuite à adapter le modèle au lieu.

## 1. Matériel (comme pour le DATASET)

- Émetteur S (`csi_send`), point d'accès `wisee_ap`, 4 récepteurs `csi_recv_udp` (RX1 à RX4) : voir `acquisition/firmware/README_UDP.md`.
- Récepteurs **dans les coins** de la zone, émetteur S et AP **au centre**, si possible à des hauteurs différentes (0,3 m à 1,8 m).
- Zone dégagée d'environ 3 × 3 m minimum, une **chaise**, et un **matelas ou un tapis épais** pour les chutes.
- Noter les dimensions de la pièce et les positions des cartes (photo + mètre) : à mettre dans `consignes.json` ou à nous envoyer.

## 2. Installation (une fois)

```
cd CSI_Prep_PFE
python -m venv venv
venv\Scripts\pip install numpy scipy pandas scikit-learn matplotlib pyserial
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. Lancer une session

PC connecté au Wi-Fi `WISEE_CSI`, puis :

```
venv\Scripts\python.exe -m skeleton.session --udp 5005 --map 192.168.4.201=RX1 192.168.4.202=RX2 192.168.4.203=RX3 192.168.4.204=RX4 --lieu maison_X --personne sujet1 --protocole court
```

- `--protocole court` (≈ 2 min) pour le premier essai, puis `--protocole standard` (≈ 4 min 30).
- L'écran (et un bip) annonce chaque consigne : **SORTEZ DE LA ZONE**, **RESTEZ DEBOUT**, **ASSEYEZ-VOUS**, **MARCHEZ**, **TOMBEZ (sur le matelas) puis RESTEZ AU SOL**.
- Pendant la session, la ligne du bas montre la consigne, l'état détecté par le modèle et le nombre de trames reçues par récepteur.
- Pour « sortez de la zone » : vraiment sortir de la zone des capteurs (l'opérateur reste au PC, toujours à la même place).
- La personne qui suit les consignes ne doit pas être celle qui tape au clavier (ou alors lancer, puis aller dans la zone pendant les 10 s d'attente).

Chaque session crée `acquisition/data/MAISON/session_AAAAMMJJ_HHMMSS/` (CSV des 4 RX, `consignes.json`, `predictions.csv`, `bilan.json`).

## 4. Combien de sessions

| Objectif | Sessions |
|---|---|
| Juste tester le modèle actuel dans ce lieu | 1-2 « court » |
| Adapter le modèle au lieu | **4 à 6 « standard »**, en changeant l'endroit où l'on marche / s'assoit / tombe, idéalement 2 personnes |
| Chutes supplémentaires | quelques `--protocole chutes` (sur matelas) |

Vérifier dans le bilan que les 4 récepteurs envoient **≥ 30 trames/s** ; sinon corriger avant de continuer (Wi-Fi, alimentation, `--map`).

## 5. Adapter le modèle (sur n'importe quel PC du projet)

Envoyer le dossier `acquisition/data/MAISON/` (sans les `.txt`), puis :

```
venv\Scripts\python.exe -m skeleton.finetune acquisition/data/MAISON --nom maison_X
```

Avec au moins 2 sessions, le script affiche le score **avant / après** sur une session laissée de côté (évaluation honnête),
puis enregistre `skeleton/checkpoints/skeleton_tcn_maison_X.pt`. Pour l'utiliser en direct :

```
venv\Scripts\python.exe -m skeleton.session ... --ckpt skeleton/checkpoints/skeleton_tcn_maison_X.pt
```
