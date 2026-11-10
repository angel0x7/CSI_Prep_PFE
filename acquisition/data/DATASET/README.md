# Dataset WISEE – campagne du 2026-11-10

Enregistrements CSI de la séance du 10/11/2026 : 4 récepteurs ESP32-S3 (RX1 à RX4) en UDP, émetteur S,
prises de **8 s** (20 s pour `vide`). Rangés et renommés le soir même.

## Arborescence

```
DATASET/
├── README.md
├── index.csv                         une ligne par prise : date, fps et pertes par RX, qualité, ancien nom
├── correspondance_anciens_noms.csv   nouveau chemin ↔ ancien chemin
└── <activité>/p<nombre de personnes>/prise_<NNN>/
        RX1.csv … RX4.csv   format équipe IA (129 colonnes : timestamp_ms + 64 sous-porteuses LLTF im/re)
        RX1.txt … RX4.txt   bruts (non versionnés sur git)
        summary.json        date, note, trames/s, pertes, synchronisation des RX
```

Les prises sont numérotées dans l'ordre chronologique, à l'intérieur de chaque dossier `p<N>`.

## Contenu

| Activité | Personnes | Prises | Description |
|---|---|---|---|
| `chute` | p1 | 50 | une personne debout, immobile, qui chute vers 2 s puis reste au sol |
| `assis` | p1 | 50 | une personne assise immobile |
| `assis` | p5 | 15 | 5 personnes assises immobiles |
| `debout` | p1 | 50 | une personne debout immobile |
| `debout` | p2 / p4 / p5 | 20 / 20 / 20 | 2, 4 ou 5 personnes debout immobiles |
| `marche` | p1 / p2 / p4 / p5 | 15 / 16 / 16 / 20 | 1, 2, 4 ou 5 personnes qui marchent (allers-retours) |
| `vide` | p0 | 2 | pièce vide (toutes deux inutilisables, voir plus bas) |

Total : **294 prises**. Les prises `p1` ont été faites avec le sujet 1 (à confirmer si la même personne pour chute, assis, debout et marche).

## Qualité : à lire avant d'utiliser les données

Pendant toute la séance, une partie des trames a été perdue entre les récepteurs et le PC (Wi-Fi) :
les pertes vont de 11 à 96 % selon les prises, alors qu'elles étaient d'environ 0,5 % lors de l'essai du 04/11 à 3 récepteurs.
Les trames perdues ne sont pas les mêmes d'un récepteur à l'autre.

La colonne `qualite` de `index.csv` :
- `ok` : pertes ≤ 5 % sur tous les RX — **aucune prise de cette séance** ;
- `degradee` : au moins un RX au-dessus de 5 % de pertes (269 prises) ;
- `inutilisable` : au moins un RX avec moins de 100 trames sur la prise, soit moins de ~12 trames/s (25 prises, dont les 2 `vide`).

| Dossier | Médiane des pertes max par prise |
|---|---|
| chute/p1 | 58 % |
| assis/p1, p5 | 62 %, 44 % |
| debout/p1, p2, p4, p5 | 53 %, 45 %, 27 %, 38 % |
| marche/p1, p2, p4, p5 | 31 %, 41 %, 36 %, 23 % |

Les `timestamp_ms` des CSV restent justes (horloge de l'ESP) : les trous dus aux pertes se voient dans le temps,
on peut rééchantillonner ou écarter les prises trop trouées.

## Montage

- 4 récepteurs dans les coins de la pièce, émetteur S et point d'accès au centre (positions exactes à compléter).
- L'opérateur reste au PC, au même endroit, pendant toutes les prises.

## Journal

| Date | Remarques |
|---|---|
| 2026-11-10 | Séance complète (294 prises). Pertes Wi-Fi élevées toute la séance, déjà présentes au test de début (7-41 %) ; elles augmentent en milieu de séance (assis/p1 prises 38 à 50 : 70-94 %). Les 2 prises `vide` sont inutilisables (0 trame pour la 1re). Problème à résoudre avant la prochaine campagne. |
| 2026-11-10 (soir) | Tri et renommage : `EMPTY/s00` → `vide/p0`, `chutes/s01` → `chute/p1`, `assis/s01` → `assis/p1`, `assis_plusieurs` → `assis/pN`, `debout/s01` → `debout/p1`, `immobile_plusieurs` (debout) → `debout/pN`, `marche_plusieurs` → `marche/pN`. Ancien index conservé dans `data/index_DATASET_avant_tri_20261110.csv`. |
