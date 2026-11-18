# -*- coding: utf-8 -*-
"""
Configuration commune du module squelette (géométrie de la salle, signal, classes).

!!! Les positions ci-dessous sont des HYPOTHÈSES (salle de classe ~30 places,
RX dans les coins, émetteur S + AP au centre). Les remplacer par les mesures
réelles dès qu'elles sont connues : le simulateur et l'inférence les utilisent.
"""
import numpy as np

# --------------------------------------------------------------------------
# Salle (mètres). Repère : x = longueur, y = largeur, z = hauteur, origine = coin RX1.
# --------------------------------------------------------------------------
ROOM_L, ROOM_W, ROOM_H = 9.0, 7.0, 3.0

RX_POS = {                       # récepteurs ESP32-S3 dans les coins, posés à ~1 m
    "RX1": (0.3, 0.3, 1.0),
    "RX2": (8.7, 0.3, 1.0),
    "RX3": (8.7, 6.7, 1.0),
    "RX4": (0.3, 6.7, 1.0),
}
TX_POS = (4.5, 3.5, 1.0)         # émetteur S (ESP-NOW ~90 trames/s), au centre
RX_NAMES = list(RX_POS)

# --------------------------------------------------------------------------
# Signal Wi-Fi / format CSV
# --------------------------------------------------------------------------
FC = 2.462e9                     # canal 11
DF = 312.5e3                     # espacement des sous-porteuses
C = 3e8
TX_RATE_HZ = 90.0                # trames émises par S
# Colonnes complexes du CSV (ordre ESP32 standard 0..31, -32..-1) : on garde 1..26 et -26..-1
SC_COLS = np.r_[1:27, 38:64]
SC_FREQ_IDX = np.r_[1:27, -26:0]  # indice fréquentiel signé correspondant
N_SC = len(SC_COLS)              # 52
SC_POOL = 2                      # moyenne de sous-porteuses voisines -> 26 bandes
N_BANDS = N_SC // SC_POOL

# --------------------------------------------------------------------------
# Grille temporelle des features
# --------------------------------------------------------------------------
FS = 20.0                        # Hz (bins de 50 ms : absorbe les pertes et la désynchro RX)
TAKE_S = 8.0                     # durée d'une prise du DATASET
WIN_S = 4.0                      # fenêtre d'entraînement
FEAT_PER_RX = 2 * N_BANDS + 2    # dyn(26) + static(26) + niveau(1) + masque(1)
N_IN = FEAT_PER_RX * len(RX_NAMES)

# --------------------------------------------------------------------------
# Classes (groupes d'activité) et "atomes de pose" du décodeur
# --------------------------------------------------------------------------
GROUPS = ["vide", "debout", "assis", "marche", "chute", "au_sol"]
N_WALK_ATOMS = 12                # phases du cycle de marche
N_FALL_ATOMS = 8                 # avancement de la chute
ATOM_GROUP = (["vide", "debout", "assis"]
              + ["marche"] * N_WALK_ATOMS
              + ["chute"] * N_FALL_ATOMS
              + ["au_sol"])
N_ATOMS = len(ATOM_GROUP)        # 24
ATOM_GROUP_IDX = np.array([GROUPS.index(g) for g in ATOM_GROUP])

NOMINAL_HEIGHT = 1.75
