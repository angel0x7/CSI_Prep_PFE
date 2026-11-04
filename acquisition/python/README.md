# WISEE – chaîne d'acquisition et d'analyse CSI (v0.1)

Première brique logicielle de WISEE. Elle reçoit le flux CSI d'un ESP32-S3, le contrôle, l'enregistre et l'analyse.
Elle fonctionne **dès aujourd'hui sans matériel** grâce à un simulateur. Le jour où les cartes arrivent,
on remplace le simulateur par le vrai ESP32 sans rien changer d'autre.

```
 [ESP32-S3 csi_recv] --série 2000000--\
                                      >--> record_csi.py --> data/session_*.txt --> analyze_session.py --> reports/
 [simulate_esp32.py] -----UDP--------/                                                    (figures + rapport)
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `wisee_csi/parser.py` | Décode les lignes `CSI_DATA` du firmware officiel **espressif/esp-csi**. Gère le format ESP32/S3 (25 champs) et le format C5/C6 (15 champs), et la convention [imaginaire, réel] par sous-porteuse |
| `wisee_csi/metrics.py` | Indicateurs de qualité : trames/s, pertes, doublons et désordre (via le compteur `id` de l'émetteur), régularité, qualité par sous-porteuse, indice de mouvement |
| `wisee_csi/channel_model.py` | Modèle de canal **synthétique** du simulateur : trajets multiples, personne (marche, respiration, chute), porteuses nulles, bruit, quantification |
| `simulate_esp32.py` | Faux ESP32-S3. Envoie en UDP des lignes au **format exact** de `csi_recv` (HT40, 192 sous-porteuses, 100 trames/s) selon un scénario, et écrit la vérité terrain |
| `record_csi.py` | Enregistreur depuis l'UDP, le port série (vrai ESP32) ou un fichier. Statistiques en direct, session horodatée et résumé JSON |
| `analyze_session.py` | Analyse hors ligne. Produit 5 figures et un `report.md` chiffré |
| `tests/test_parser.py` | Tests unitaires, dont un sur la ligne d'exemple officielle du README d'esp-csi |

## Installation (Windows, PowerShell) : **Python 3.12** pour tout le groupe

```
cd <depot>\acquisition\python
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover tests        # doit afficher OK
```

## Démo sans matériel (2 terminaux)

Terminal 1, l'enregistreur :
```
python record_csi.py --udp 5005 --duration 65 --note "test simulateur"
```
Terminal 2, le faux ESP32 (1 % de pertes et 2 ms de gigue pour faire réaliste) :
```
python simulate_esp32.py --scenario "empty:8,walk:8,sit:20,fall:2,lying:20" --loss 0.01 --jitter-ms 2
```
Puis on analyse :
```
python analyze_session.py data\session_XXXXXXXX_XXXXXX.txt --labels data\sim_labels.csv
```
Les figures et le rapport sont dans `reports\session_…\`. 

Activités disponibles dans le scénario : `empty`, `stand`, `sit`, `walk`, `fall`, `lying`.

## Avec un vrai ESP32-S3

1. Flasher `esp-csi/examples/get-started/csi_send` sur une carte et `csi_recv` sur l'autre (ESP-IDF **5.5.5**).
2. Fermer `idf.py monitor`, puis lancer :
```
python record_csi.py --serial COM6 --duration 300 --note "piece vide, salle XXX, emetteur a 3 m"
python analyze_session.py data\session_XXXXXXXX_XXXXXX.txt
```
Les lignes de log d'ESP-IDF (`I (…) csi_recv: …`) sont ignorées par le parseur.

## Ce que montre l'exemple de simulation (`reports/simulation_avant_materiel/report.md`)

- L'enregistreur détecte **exactement** les pertes injectées : 54 trames perdues par le simulateur, 54 comptées, soit 0,93 %.
- Il repère automatiquement les **14 sous-porteuses à masquer** (DC et bandes de garde) parmi les 128 du HT-LTF.
- L'indice de mouvement sépare nettement la marche et la chute (indice moyen de 0,11 à 0,18) de l'immobilité (environ 0,03).
- Sur les phases immobiles, la respiration simulée (15/min) est retrouvée : **15,0 respirations/min**.

**Limite importante** : ces chiffres valident **le logiciel**, pas la détection réelle. Le canal est synthétique,
et la personne immobile y est presque indiscernable de la pièce vide avec cet indice simple. Les vraies
performances se mesureront sur les enregistrements ESP32 avec le même script.

## Limites connues (v0.1)

- Le simulateur ne modélise ni le bruit réel, ni les interférences, ni les variations du gain automatique, ni plusieurs personnes.
- L'indice de mouvement est volontairement simple : c'est une référence, pas le futur modèle de détection.
- Le firmware `csi_recv` envoie sur le port série. L'envoi en UDP depuis l'ESP32 reste à écrire.
- Il n'y a pas encore d'affichage temps réel : l'analyse se fait hors ligne.

## Démo de la soutenance intermédiaire (dossier `demo/`)

Tout est dans `demo/` : le script `demo_soutenance.py`, les lanceurs `LANCER_DEMO.bat` et `REFAIRE_ANALYSE.bat`,
les dossiers de données et de résultats, et **`demo/README_DEMO.md`** qui explique toute la démarche.

`record_multi.py` accepte maintenant `--name` (nom du dossier de session, jamais écrasé).
