# Méthodologie de la campagne de mesures
## Localisation · détection de chute · squelette · respiration et rythme cardiaque

Objectif de la campagne : obtenir des données **avec vérité terrain** pour entraîner et surtout **évaluer honnêtement**
ce que nos 4 ESP32-S3 permettent réellement. Chaque objectif a une mesure de référence, un protocole,
et un critère de réussite **fixé avant** de mesurer.

---

## 1. Point de départ : ce que permettent les données actuelles

| Objectif | Avec le DATASET du 10/11 | Pourquoi | Ce qui manque |
|---|---|---|---|
| Localisation | **impossible à entraîner ou à évaluer** | aucune position enregistrée ; test L2/L3 du 14/10 non concluant | position de la personne à chaque instant |
| Détection de chute | possible et évaluée (50/50 chutes, 0 fausse alerte sur montages) | — | vraies transitions enregistrées d'un trait, gestes qui ressemblent à une chute, autre lieu |
| Squelette | **seulement paramétrique** (piloté par l'activité) | aucune position d'articulation enregistrée | position des articulations à chaque instant (caméra) |
| Respiration | non évaluée | pas de mesure de référence | rythme respiratoire de référence |
| Rythme cardiaque | non évalué | pas de mesure de référence | rythme cardiaque de référence |

Ce qui est en simulation (`skeleton/results/faisabilite*.json`, `localisation_sim.json`) n'est qu'une **indication** : seules
les mesures ci-dessous permettront de conclure.

---

## 2. Prérequis communs (à valider AVANT chaque séance)

**Acquisition**
- Pertes **< 5 %** et **≥ 80 trames/s** sur chaque récepteur, vérifiées sur 2 minutes (bilan de `skeleton.session`).
  Pistes si ce n'est pas le cas : liaison série USB (testée à 2 Mbauds le 14/10 : 1,5 % de pertes), format UDP binaire compact,
  point d'accès sur un autre canal que la mesure.
- **Garder les `.txt` bruts** (HT-LTF 40 MHz, 128 sous-porteuses, numéro de trame de l'émetteur) : les copier sur un disque après chaque séance.
- Un **seul PC** enregistre tout (CSI, webcam, consignes) : une seule horloge.

**Montage figé et documenté**
- Position (x, y, z) de chaque carte au mètre ruban, orientation des antennes, photo + schéma coté. Cartes **fixées au scotch** ;
  rien ne bouge pendant toute la campagne. Si on bouge, c'est un nouveau « montage », noté comme tel.
- L'opérateur au PC reste **toujours au même endroit**, hors de la zone, noté sur le schéma.

**Sécurité et éthique**
- Chutes **uniquement sur un matelas épais** (gym ou matelas de lit), personne en bonne santé, arrêt à la moindre gêne.
- Vidéo de personnes : **accord écrit** de chaque participant, vidéos gardées en local (jamais publiées, pas sur git),
  utilisées uniquement pour étiqueter les données. Le système final reste sans caméra.

---

## 3. Montage de la zone de mesure

```
        RX4 (h=1,5 m) ●─────────────── 4 m ───────────────● RX3 (h=0,5 m)
                      │  A4     B4     C4     D4           │
                      │  A3     B3     C3     D3           │     webcam en hauteur
                      │  A2     B2     C2     D2           │     dans un coin (voit
                      │  A1     B1     C1     D1           │     toute la grille)
        RX1 (h=0,5 m) ●───────────────────────────────────● RX2 (h=1,5 m)
                         émetteur S + AP : au centre (h=1 m) ou au milieu d'un bord
```

- Zone d'environ **4 × 4 m** (pas toute la salle), **grille au sol au scotch**, cases de 1 m nommées A1 à D4.
- Récepteurs **à des hauteurs différentes** (alternance 0,5 m / 1,5 m) : donne de l'information sur la hauteur du corps.
- Option firmware à décider après la séance pilote : toutes les cartes émettent à tour de rôle (liens qui se croisent).

---

## 4. Mesures de référence (vérité terrain)

| Objectif | Référence | Matériel | Coût |
|---|---|---|---|
| Localisation | case de la grille (consigne) + **position (x, y) des pieds** par webcam | webcam USB fixe en hauteur, grille calibrée (4 coins connus) | ~30 € |
| Chute | consigne + **instant exact** sur la vidéo | même webcam, matelas | — |
| Squelette | **17 points COCO** extraits de la vidéo (YOLOv8-pose ou MediaPipe), 3D approximative possible | même webcam (une 2e webcam de côté si possible) | ~30 € |
| Respiration | **accéléromètre d'un téléphone posé sur le thorax** (appli d'enregistrement de capteurs, ex. phyphox) + respiration au **métronome** | téléphone, métronome | 0 € |
| Rythme cardiaque | **oxymètre de pouls** ou ceinture cardio (relevé continu si possible) | oxymètre ~20 € / ceinture ~60 € | 20–60 € |

**Synchronisation** : au début de chaque session, **clap visible** (vidéo) + **3 tapes sur le téléphone** (pic dans l'accéléromètre),
au moment annoncé par le script. Tous les fichiers sont recalés sur cet instant.

---

## 5. Protocoles par objectif

Chaque session commence et finit par **60 s de pièce vide** (référence et mesure de la dérive).
Les consignes sont annoncées par `skeleton.session` (texte + bip), dans un **ordre aléatoire** quand c'est possible.

### 5.1 Localisation (≈ 30 min par personne)

| Bloc | Contenu | Durée |
|---|---|---|
| Statique | chaque case A1→D4 **dans un ordre aléatoire** : debout 15 s, puis assis 15 s (chaise déplacée avec la personne) | 16 × 30 s = 8 min |
| Statique, répétition | même chose, autre ordre aléatoire | 8 min |
| Trajets | marche le long des 4 lignes, des 4 colonnes et des 2 diagonales, 2 allers-retours chacun | ~6 min |
| Marche libre | déplacements naturels dans la zone (vidéo = vérité) | 3 min |

### 5.2 Détection de chute (≈ 25 min par personne)

| Bloc | Contenu | Répétitions |
|---|---|---|
| Chutes | sur matelas, **vers l'avant, l'arrière, le côté**, depuis : debout immobile, en marchant, juste après s'être levé d'une chaise. Chaque chute suivie de 10 s au sol puis se relever | ≥ 10 chutes / personne |
| **Faux amis** (ne sont PAS des chutes) | s'asseoir brusquement, se baisser ramasser un objet, s'allonger volontairement et lentement, s'accroupir, sauter, laisser tomber un sac lourd sur le matelas | 3 × chaque geste |
| Activité normale longue | 10 min de vie normale dans la zone (marcher, s'asseoir, se lever, téléphoner...) | 1 fois |

L'activité normale longue sert à mesurer le **nombre de fausses alertes par heure**, le critère qui compte en pratique.

### 5.3 Squelette (≈ 20 min par personne, avec vidéo obligatoire)

| Bloc | Contenu |
|---|---|
| Gestes des bras | lever / baisser un bras puis l'autre, les deux, bras tendus devant, sur les côtés (5 × chaque) |
| Gestes du tronc | se pencher en avant, sur les côtés, tourner le buste, s'accroupir, se relever |
| Postures | debout, assis, à genoux, allongé (dos / ventre / côté), 20 s chacune |
| Enchaînements | marcher → s'asseoir → se lever → marcher, en continu |
| Orientation | refaire les gestes des bras face à RX1, puis face à RX3 |

Ces mouvements sont **continus et variés** : c'est ce qui manquait au DATASET (5 postures figées seulement).

### 5.4 Respiration et rythme cardiaque (≈ 25 min par personne)

| Bloc | Contenu | Durée |
|---|---|---|
| Respiration guidée | assis immobile, respiration au métronome à **6, 10, 15, 20, 25 resp/min** | 5 × 90 s |
| Respiration libre | assis, puis debout, puis allongé, respiration naturelle | 3 × 90 s |
| Apnée | 15 s d'apnée au milieu d'une minute immobile | 3 fois |
| Distance aux liens | respiration à 15 resp/min sur une ligne émetteur-récepteur, puis à 1 m, puis à 2 m | 3 × 90 s |
| Cardiaque | assis très immobile 2 min (oxymètre au doigt), puis **30 flexions**, puis de nouveau 2 min assis (rythme qui varie de ~60 à ~110 bpm) | ~6 min |

**Attente honnête** : la respiration d'une personne immobile est plausible. Le rythme cardiaque avec 4 ESP32 mono-antenne
est **peu probable**. Ce protocole permet de le **démontrer dans un sens ou dans l'autre**, chiffres à l'appui.

### 5.5 En complément (si le temps le permet)
- **Comptage** : 1, 2, 3 puis 4 personnes sur des cases connues, immobiles puis en marche (3 × 2 min chaque).
- **À travers le mur** : récepteurs derrière une cloison, pièce vide / personne immobile / personne qui marche (3 × 2 min).

---

## 6. Organisation des séances

1. **Séance pilote (≈ 1 h 30, 1 personne)** : prérequis du §2, puis 4 quadrants (localisation), 5 chutes + 5 faux amis,
   10 min de gestes, 3 respirations au métronome. → Analyse (§8) **avant** de lancer la suite : le signal porte-t-il l'information ?
2. **Campagne (≈ 3 séances de 2 h)** : **3 à 5 personnes**, **au moins 2 jours différents** (porte, meubles tels qu'ils sont ce jour-là).
3. **Rôles** pendant une séance : un **opérateur** (PC, vérifie les trames/s), un **sujet**, un **observateur** qui remplit la fiche.

**Fiche de séance** (une par session) : date, heure, lieu, montage (photo), personne (taille, vêtements), porte ouverte/fermée,
personnes présentes hors zone, incidents (carte redémarrée, consigne mal suivie à t = …, chute ratée…).

---

## 7. Rangement des données

```
acquisition/data/CAMPAGNE_<date>/
├── montage.json              positions et hauteurs des cartes, orientation, schéma, photo
├── index.csv                 une ligne par session : objectif, personne, jour, durée, trames/s, qualité
└── <objectif>/<personne>/session_AAAAMMJJ_HHMMSS/
        RX1..RX4.csv          format DATASET (versionné)
        RX1..RX4.txt          bruts (NON versionnés, sauvegardés sur disque)
        consignes.json        étiquettes (segments début/fin/état/case)
        predictions.csv       sortie du modèle en direct
        video.mp4             (local, non versionné)
        reference_*.csv       accéléromètre / oxymètre, recalés sur le clap
        fiche.txt             fiche de séance
```

---

## 8. Analyse et critères de décision (fixés avant de mesurer)

**Règle d'évaluation** : on entraîne sans une personne (ou sans un jour) et on teste sur elle (ou sur lui).
Jamais de test sur la même séance que l'entraînement. Les critères ci-dessous sont des propositions, à valider avec l'encadrant.

| Objectif | Métrique | Référence de comparaison | Réussi si | Sinon |
|---|---|---|---|---|
| Localisation | % de quadrants justes, % de cases justes, erreur médiane (m) — immobile et en mouvement séparés | « toujours au centre » ; hasard (25 % / 6 %) | quadrants ≥ 80 % sur un jour non vu | zone probable + confiance, ou montage à liens croisés, ou limite démontrée |
| Chute | sensibilité (% de chutes détectées), fausses alertes par heure d'activité normale, délai | — | ≥ 90 % des chutes, ≤ 1 fausse alerte / h | réglage des seuils, plus de « faux amis » à l'entraînement |
| Squelette | erreur moyenne par articulation (mm, 2D ou 3D), PCK, sur une personne non vue | **squelette type de l'activité** (ce qu'on a aujourd'hui) | au moins 20 % meilleur que la référence | squelette paramétrique conservé, limite démontrée |
| Respiration | erreur absolue (resp/min) sur fenêtres de 30 s, sujet immobile | référence accéléromètre | erreur médiane ≤ 2 resp/min sur ≥ 80 % des fenêtres | limite démontrée (préciser la distance maximale) |
| Rythme cardiaque | erreur absolue (bpm) sur fenêtres de 30 s | oxymètre | ≤ 5 bpm sur ≥ 70 % des fenêtres | limite démontrée |

---

## 9. Outils

| Existe déjà | À préparer avant la campagne |
|---|---|
| `skeleton.session` : consignes guidées + enregistrement + modèle en direct | protocole « grille » (cases en ordre aléatoire, trajets, blocs pièce vide) |
| `skeleton.finetune` : adaptation à un nouveau lieu | enregistrement webcam synchronisé + calibration de la grille |
| `skeleton.viewer` : visualisation et test en direct | extraction des 17 points (YOLOv8-pose / MediaPipe) et de la position des pieds |
| `csi_preprocess_breathing.py` : filtrage respiration | import et recalage des références (téléphone, oxymètre) |
| | scripts d'analyse qui appliquent le §8 et sortent le verdict |

---

## 10. Risques

| Risque | Conséquence | Parade |
|---|---|---|
| Pertes Wi-Fi élevées (comme le 10/11) | données inutilisables | prérequis du §2 vérifié à chaque séance ; série USB |
| Carte déplacée ou redémarrée | empreintes faussées | cartes scotchées, montage photographié, noter dans la fiche |
| Sujet qui ne suit pas la consigne | étiquettes fausses | observateur + vidéo ; marquer l'instant dans la fiche |
| Trop peu de personnes / de jours | résultats trop optimistes | minimum 3 personnes et 2 jours |
| Blessure pendant les chutes | — | matelas épais, chutes contrôlées, arrêt immédiat en cas de gêne |
