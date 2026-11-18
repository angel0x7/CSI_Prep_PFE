# Module squelette 17 points (COCO) à partir du CSI Wi-Fi

Estimation, **à partir du seul CSI** des 4 récepteurs ESP32-S3, du squelette COCO-17 (3D, en mètres)
d'**une personne**, avec son activité (debout / assis / marche / chute / au sol / vide), en temps réel.

> **Limite à connaître (et à dire en soutenance).** Nos prises n'ont pas de caméra : aucune position
> d'articulation n'a été mesurée. Un réseau qui *régresse* 17 points ne peut donc pas être appris
> directement sur nos données. La solution retenue :
> 1. un **simulateur physique** (corps = 36 réflecteurs dans une salle de classe, matériel ESP32 reproduit,
>    calibré sur nos vraies prises) fournit des données **avec** vérité terrain squelette ;
> 2. nos **vraies prises** fournissent la supervision d'**activité** (étiquettes faibles, instant de chute estimé) ;
> 3. le modèle sort des squelettes via un **décodeur à atomes de pose** : chaque sortie est un mélange
>    de poses anatomiquement valides (+ petit résidu), donc toujours plausible.
>
> Le squelette affiché est donc **cohérent avec l'activité et la dynamique détectées**, et précis au sens
> du simulateur ; il n'est **pas** une mesure des articulations comme le ferait un système supervisé par caméra.

## Arborescence

```
skeleton/
├── config.py       géométrie de la salle (HYPOTHÈSES à remplacer par les mesures), signal, classes
├── poses.py        squelettes COCO-17 par cinématique directe (debout, assis, cycle de marche, chute, sol)
├── synth.py        simulateur CSI corps articulé -> prises au format DATASET + labels.npz
├── calibrate.py    comparaison statistiques réelles / simulées (réglage du simulateur)
├── features.py     CSV 4 RX -> features causales 20 Hz (216 canaux)
├── labels.py       étiquettes faibles des vraies prises (+ détection de l'instant de chute)
├── augment_real.py nouvelles prises générées À PARTIR DES VRAIES (x4) -> DATASET_AUG
├── dataset.py      caches d'entraînement (skeleton/cache/, non versionné)
├── model.py        SkeletonTCN (~0.25 M paramètres, causal)
├── train.py        validation croisée + ablations, modèle final
├── infer.py        inférence hors ligne (JSON) et en flux (SkeletonEstimator) pour le dashboard
├── visualize.py    figures
├── checkpoints/skeleton_tcn.pt
├── results/        résultats de validation croisée (JSON + logs)
├── figures/
└── tests/
```

Deux jeux de données générés, au même format que le vrai `DATASET` (qui n'est jamais modifié) :

| Dossier | Contenu | Rôle |
|---|---|---|
| `acquisition/data/DATASET_AUG/` | copie du DATASET réel + **4 prises augmentées par prise p1** (`prise_NNN_augK`, 596 au total) | plus de données **réelles** pour l'activité |
| `acquisition/data/DATASET_SYNTH/` | prises **simulées** + `labels.npz` (keypoints 3D par trame, position, cap) | seule source de vérité terrain **squelette** |

Les deux sont régénérables en une commande (seed fixée) et ne sont pas versionnés (taille).

**Augmentation des vraies prises** (`augment_real.py`) : étirement temporel ×0,9–1,15, inversion du temps
(sauf chute), décalage de l'instant de chute, pertes supplémentaires en rafales, gain/bruit/re-quantification
par RX, ondulation spectrale, désynchronisation des RX, et recombinaison « forme statique de la prise A ×
mouvement de la prise B » (même classe). Chaque copie garde sa prise `source` : en validation croisée,
une prise et ses copies sont toujours du même côté, et le test ne contient que des prises réelles d'origine.
L'augmentation n'ajoute pas d'information nouvelle sur la posture : elle rend le modèle robuste aux
variations (vitesse, pertes, matériel), ce qui compte avec seulement 149 prises utilisables.

## Utilisation

```bash
venv\Scripts\activate
pip install numpy scipy pandas scikit-learn matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 0. interface graphique de test (squelette 3D animé, activité, frise, énergie CSI, plan)
python -m skeleton.viewer                 # -> http://127.0.0.1:8765

# 1. inférence sur une prise (le modèle entraîné est versionné)
python -m skeleton.infer acquisition/data/DATASET/chute/p1/prise_010 --png

# 2. ré-entraîner
python -m skeleton.dataset --n-per-class 500 --k-aug 0   # caches synthétique + réel (≈3 min, 8 cœurs)
python -m skeleton.dataset --real-only --k-aug 4         # cache réel augmenté (real_aug)
python -m skeleton.train cv --steps 800 [--real-cache real_aug]   # validation croisée + ablations
python -m skeleton.train final --steps 1500              # -> checkpoints/skeleton_tcn.pt (utilise real_aug)

# 3. générer les dossiers de données (CSV lisibles)
python -m skeleton.augment_real --k 4                     # -> acquisition/data/DATASET_AUG
python -m skeleton.synth --n-per-class 40 --out acquisition/data/DATASET_SYNTH

# 4. tests
python -m unittest discover -s skeleton/tests -t .
```

En temps réel (côté serveur du dashboard) :

```python
from skeleton.infer import SkeletonEstimator
est = SkeletonEstimator()
# à chaque paquet UDP : est.push("RX2", t_ms, valeurs_v0_a_v127)
frame = est.latest(x=4.2, y=3.1, heading=0.0)   # x, y, cap : fournis par le module de localisation
frame["activity"], frame["keypoints_world"], frame["keypoints_coco"]
```

Le squelette est exprimé dans le **repère du corps** (x gauche, y avant, z haut, origine au sol entre les
hanches). La position et l'orientation dans la salle viennent de la localisation / du tracking
(l'orientation peut être prise égale à la direction de la vitesse quand la personne marche).

## Méthode

**Features** (`features.py`) : amplitude des 52 sous-porteuses utiles, normalisée trame par trame (annule
l'AGC), regroupée en 26 bandes, moyennée en bins de 50 ms (absorbe les pertes et la désynchronisation des RX,
p95 ≈ 30 ms), puis pour chaque RX : variation rapide (mouvement), forme statique (2 s passées), niveau,
masque de bins reçus. Tout est causal. La phase n'est pas utilisée (CFO/STO aléatoires sur ESP32).

**Simulateur** (`synth.py`) : salle 9 × 7 × 3 m, murs (sources images), 30 tables + 30 chaises,
opérateur assis au PC ; personne = 17 keypoints + 19 milieux d'os pondérés (tronc fort, extrémités faibles) ;
ombrage du trajet direct dans la zone de Fresnel ; respiration, balancement, petits gestes ; matériel :
réponse en fréquence propre à chaque RX, AGC partielle, bruit, int8, pertes en rafales (Gilbert-Elliott)
tirées de la distribution réelle de `index.csv`. Les réglages ont été faits avec `calibrate.py`
pour que l'énergie de mouvement, le relief spectral et la variation de niveau des prises simulées
soient dans les mêmes ordres de grandeur que les vraies, classe par classe.

**Étiquettes réelles** (`labels.py`) : l'activité de la prise ; pour `chute`, l'instant de chute est
estimé (rupture de la forme du canal × pic de mouvement). Validé sur simulation : erreur médiane 0,3 s,
5/30 erreurs > 1 s ; les bins à ±0,4 s des transitions sont ignorés dans la perte.

**Modèle** (`model.py`) : TCN causal (5 blocs dilatés, champ réceptif 3,1 s), têtes : 24 atomes de pose,
taille, résidu ±15 cm. Pertes : simulé = atome + activité + MPJPE + taille ; réel = activité seulement.

Résultats de validation croisée : `skeleton/results/cv_*.json`.
