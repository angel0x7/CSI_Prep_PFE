# Évaluation des résultats

Ce document explique comment lire les résultats générés par les scripts d’évaluation du projet et ce qu’ils signifient en pratique pour le prétraitement des données CSI.

---

## 1. Objectif des évaluations

Les scripts d’évaluation comparent plusieurs variantes de prétraitement sur les mêmes données d’activité humaine.

Les variantes sont :

- A brut : données sans filtrage passe-bande, sans correction Hampel
- B Hampel : suppression des valeurs aberrantes uniquement
- C complet : pipeline complet avec Hampel + filtrage Butterworth
- D passe-bande : version avec filtrage passe-bande, sans Hampel (dans l’évaluation de fichier)

L’idée est de mesurer si le prétraitement améliore réellement la qualité des données pour la classification des activités.

---

## 2. Scripts d’évaluation présents

### `eval_pipeline.py`

Ce script extrait des fenêtres temporelles à partir des fichiers `.npz` générés par le pipeline de prétraitement, puis applique des modèles de classification.

Il compare :

- Random Forest
- Logistic Regression

en validation croisée par sujet.

### `eval_file.py`

Ce script applique une variante plus orientée “fichiers / sujets” :

- agrégation des features par individu/sujet
- évaluation par classifieur
- comparaison entre variantes
- bootstrap pour estimer la différence d’accuracy entre méthodes

---

## 3. Métriques utilisées

Les résultats affichent principalement :

- `acc` : accuracy globale
- `F1macro` : score F1 moyen sur les classes
- `AUC(chute vs reste)` : capacité de distinguer la classe chute des autres activités
- `confusion_matrix` : matrice de confusion

### Interprétation rapide

- `accuracy` indique le taux global de bonnes prédictions
- `F1macro` est utile quand les classes sont déséquilibrées
- `AUC` est important pour la détection de situations critiques comme la chute
- la matrice de confusion montre où le modèle confond les classes

---

## 4. Résultat observé sur `eval_pipeline.py`

Les résultats obtenus sont les suivants :

```text
A brut: 16510 fenêtres, 22 sujets, 20 features
   RandomForest  acc=0.353  F1macro=0.344  AUC(chute vs reste)=0.679
   LogReg        acc=0.348  F1macro=0.327  AUC(chute vs reste)=0.683
B Hampel: 16510 fenêtres, 22 sujets, 20 features
   RandomForest  acc=0.350  F1macro=0.341  AUC(chute vs reste)=0.680
   LogReg        acc=0.367  F1macro=0.346  AUC(chute vs reste)=0.685
C complet: 16510 fenêtres, 22 sujets, 20 features
   RandomForest  acc=0.368  F1macro=0.357  AUC(chute vs reste)=0.705
   LogReg        acc=0.386  F1macro=0.358  AUC(chute vs reste)=0.698
```

### Analyse

- La variante `C complet` est la meilleure parmi les trois variantes comparées ici.
- L’amélioration est particulièrement visible en AUC chute :
  - A brut : 0.679
  - C complet : 0.705
- Le score F1 macro augmente aussi :
  - A brut : 0.344
  - C complet : 0.357

### Conclusion partielle

Le pipeline complet semble améliorer la séparabilité des activités, même si les scores restent modérés. Cela montre que le prétraitement aide, mais que le problème reste difficile à traiter avec des features simples.

---

## 5. Résultat observé sur `eval_file.py`

Les résultats obtenus sont :

```text
A brut     fichiers=1647 acc=0.506 F1=0.495 AUC(chute)=0.822
B Hampel   fichiers=1647 acc=0.485 F1=0.475 AUC(chute)=0.795
C complet  fichiers=1647 acc=0.500 F1=0.491 AUC(chute)=0.813
D passe-bande fichiers=1647 acc=0.520 F1=0.511 AUC(chute)=0.844
```

### Analyse

- La meilleure performance est obtenue par `D passe-bande` :
  - accuracy = 0.520
  - F1 = 0.511
  - AUC chute = 0.844
- La variante `A brut` est également bonne : 0.506 / 0.495 / 0.822
- La variante `C complet` est un peu moins forte ici que `A brut` et `D passe-bande`
- `B Hampel` est la moins bonne sur cette exécution

### Interprétation

Cela suggère que :

- le filtrage passe-bande peut être très utile dans certaines configurations
- le Hampel n’apporte pas toujours un gain net dans cette configuration précise
- les performances dépendent fortement des features choisies, du découpage temporel et de la validation utilisée

---

## 6. Matrice de confusion

La matrice de confusion montre où le modèle se trompe.

```text
[[1190  555  804  484  240]
 [ 761  803  626  714  393]
 [ 918  729  694  623  336]
 [ 799  691  627  787  428]
 [  65  276  135  225 2607]]
```

### Exemple de lecture

Les lignes représentent les vraies classes et les colonnes les classes prédites.

Par exemple :

- la classe `WA` est généralement bien reconnue : beaucoup de vrais `WA` sont prédits comme `WA`
- les classes `FA`, `GU`, `LD`, `SD` sont plus difficiles à distinguer
- le modèle confond souvent plusieurs activités entre elles

Cela indique que les données sont relativement difficiles à séparer avec les features simples utilisées ici.

---

## 7. Analyse des écarts entre variantes

Le script affiche également des différences de précision avec intervalles de confiance bootstrap :

```text
Δacc D passe-bande - A brut: +1.3 pts  IC95% [-1.0, +3.8]
Δacc D passe-bande - C complet: +2.0 pts  IC95% [-0.3, +4.4]
Δacc C complet - A brut: -0.7 pts  IC95% [-3.1, +1.5]
Δacc B Hampel - A brut: -2.2 pts  IC95% [-4.1, -0.2]
```

### Lecture

- `D passe-bande - A brut` : +1.3 points, mais l’intervalle inclut 0
- `D passe-bande - C complet` : +2.0 points, intervalle presque à 0
- `C complet - A brut` : légèrement négatif
- `B Hampel - A brut` : la différence est négative et l’intervalle n’inclut pas 0, donc il y a un effet probablement défavorable du Hampel seul dans cette configuration

### Ce que cela veut dire

Les améliorations ne sont pas toujours statistiquement très fortes. Donc, même si certaines courbes semblent meilleures, il faut rester prudent avant de conclure à un gain robuste.

---

## 8. Ce que cela implique pour le projet

### Conclusion pratique

- le pipeline de prétraitement est utile, mais il faut choisir les paramètres avec soin
- le filtre passe-bande semble intéressant pour le signal de mouvement humain
- le Hampel n’est pas forcément bénéfique à lui seul dans cette série de données
- il faut tester plusieurs combinaisons de features et de filtres avant de trancher

### Interprétation métier

Le projet vise probablement à détecter :

- activité humaine
- posture
- chute
- respiration

Les résultats montrent que les classes les plus faciles à séparer sont celles qui ont des signatures distinctes, tandis que les classes proches restent difficiles à distinguer.

---

## 9. Conclusion générale

Les évaluations montrent que :

- les scores ne sont pas excellents, mais ils sont cohérents avec un problème de classification de gestes/postures sur des signaux CSI bruités
- la meilleure variante dépend du type de benchmark et des features utilisées
- la sélection du prétraitement est un point important mais n’est pas suffisante à elle seule pour atteindre des performances très élevées

Le point clé est que les résultats indiquent un comportement utile du pipeline, sans permettre de conclure qu’une seule variante est universellement supérieure.

---

## 10. Recommandation

Pour le projet, la stratégie la plus raisonnable est :

1. garder un pipeline robuste de nettoyage
2. utiliser un filtrage adapté à la fréquence d’intérêt
3. tester plusieurs variantes de features
4. comparer les résultats avec validation par sujet
5. utiliser la matrice de confusion pour identifier les classes qui se mélangent

Cela permet d’éviter de se baser uniquement sur la précision brute sans tenir compte du contexte des classes et des sujets.

---

