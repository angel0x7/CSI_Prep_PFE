# -*- coding: utf-8 -*-
"""
Modele de canal Wi-Fi SYNTHETIQUE pour le simulateur WISEE.

Ce n'est PAS une simulation physique fidele d'une piece : c'est un generateur
plausible qui reproduit les phenomenes utiles pour tester la chaine logicielle :
  - trajets multiples statiques (murs, meubles) -> forme en frequence stable
  - un trajet "humain" dont la longueur varie avec le mouvement
    (marche = variations rapides, respiration = petite modulation periodique,
     chute = changement brutal puis immobilite)
  - sous-porteuses nulles (DC, bandes de garde) comme sur le vrai materiel
  - bruit de mesure + quantification entiere (int8) comme le buffer ESP32

H(f, t) = sum_p a_p exp(-j 2 pi (fc + f) tau_p)  +  a_h(t) exp(-j 2 pi (fc + f) d_h(t) / c)
"""

import numpy as np

C = 3e8
FC = 2.462e9                 # canal 11 (celui utilise par esp-csi get-started)
DF = 312.5e3                 # espacement des sous-porteuses OFDM

# Ordre des sous-porteuses dans le buffer ESP32 : indices positifs puis negatifs.
LLTF_IDX = np.r_[0:32, -32:0]            # 64 sous-porteuses (20 MHz)
HT40_IDX = np.r_[0:64, -64:0]            # 128 sous-porteuses (40 MHz)


def _null_mask(idx, n_used):
    """True = sous-porteuse transmise ; False = nulle (DC / garde)."""
    a = np.abs(idx)
    if n_used == 26:                     # LLTF : -26..-1, 1..26
        return (a >= 1) & (a <= 26)
    return (a >= 2) & (a <= 58)          # HT40 : -58..-2, 2..58


LLTF_MASK = _null_mask(LLTF_IDX, 26)
HT40_MASK = _null_mask(HT40_IDX, 58)

STATES = ("empty", "stand", "sit", "walk", "fall", "lying")


class Room(object):
    """Une piece fictive : trajets statiques tires au hasard une fois pour toutes."""

    def __init__(self, seed=0, n_static=6):
        rng = np.random.default_rng(seed)
        self.tau = np.sort(rng.uniform(8e-9, 60e-9, n_static))          # retards (s)
        self.amp = rng.uniform(0.3, 1.0, n_static) * np.exp(-self.tau / 40e-9)
        self.amp[0] = 1.0                                                  # trajet direct
        self.phi = rng.uniform(0, 2 * np.pi, n_static)
        self.rng = rng

    def response(self, freqs, human_amp, human_dist):
        f = FC + freqs
        H = (self.amp[None, :] * np.exp(1j * self.phi[None, :])
             * np.exp(-2j * np.pi * f[:, None] * self.tau[None, :])).sum(axis=1)
        if human_amp > 0:
            H = H + human_amp * np.exp(-2j * np.pi * f * human_dist / C)
        return H


class HumanTrajectory(object):
    """Etat de la personne -> (amplitude du trajet reflechi, longueur du trajet en m)."""

    BREATH_HZ = 0.25          # 15 respirations / min
    BREATH_M = 0.004          # deplacement thoracique ~4 mm

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.d_walk = 6.0
        self.walk_dir = 1.0
        self.fall_start = None

    def sample(self, state, t, t_in_state, dt):
        breath = self.BREATH_M * np.sin(2 * np.pi * self.BREATH_HZ * t)
        sway = 0.002 * self.rng.standard_normal()
        if state == "empty":
            return 0.0, 0.0
        if state == "stand":
            return 0.45, 5.0 + breath + sway
        if state == "sit":
            return 0.35, 5.6 + breath + 0.5 * sway
        if state == "lying":
            return 0.25, 7.2 + breath
        if state == "walk":                                  # aller-retour a ~1 m/s
            self.d_walk += self.walk_dir * 1.0 * dt + 0.01 * self.rng.standard_normal()
            if self.d_walk > 8.0 or self.d_walk < 4.0:
                self.walk_dir *= -1.0
            return 0.45 + 0.05 * np.sin(2 * np.pi * 1.8 * t), self.d_walk
        if state == "fall":                                  # 5.0 m -> 7.2 m en ~0.8 s
            x = min(1.0, t_in_state / 0.8)
            return 0.45 - 0.2 * x, 5.0 + 2.2 * x ** 2 + 0.02 * self.rng.standard_normal()
        raise ValueError("etat inconnu : %s" % state)


def csi_buffer(room, human_amp, human_dist, rng, target_amp=40.0, noise_lsb=1.0):
    """Construit le buffer brut ESP32-S3 (HT40) : LLTF (64 sc) + HT-LTF (128 sc),
    entrelace [imag, reel] et quantifie en entiers comme le firmware."""
    parts = []
    for idx, mask in ((LLTF_IDX, LLTF_MASK), (HT40_IDX, HT40_MASK)):
        H = room.response(idx * DF, human_amp, human_dist)
        H = H / np.mean(np.abs(H[mask])) * target_amp           # "AGC" : niveau constant
        H = H + noise_lsb * (rng.standard_normal(H.size) + 1j * rng.standard_normal(H.size))
        H[~mask] = 0
        parts.append(H)
    H = np.concatenate(parts)
    raw = np.empty(2 * H.size, dtype=np.int16)
    raw[0::2] = np.clip(np.round(H.imag), -127, 127)
    raw[1::2] = np.clip(np.round(H.real), -127, 127)
    return raw
