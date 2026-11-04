# -*- coding: utf-8 -*-
"""
Indicateurs de qualite du flux CSI (cf. cahier des charges WISEE, points de vigilance) :
  - trames par seconde reellement recues
  - pertes / doublons / desordre, detectes grace au compteur 'id' envoye par l'emetteur
  - irregularite temporelle (intervalle entre trames)
  - qualite par sous-porteuse (sous-porteuses nulles ou instables a masquer)
"""

import numpy as np


class StreamStats(object):
    """Statistiques cumulees + fenetre glissante d'une seconde, mises a jour trame par trame."""

    def __init__(self):
        self.n_frames = 0
        self.n_lost = 0
        self.n_dup = 0
        self.n_reordered = 0
        self.n_bad_lines = 0
        self.last_seq = None
        self.first_ts = None
        self.last_ts = None
        self._window = []           # horodatages PC de la derniere seconde
        self.intervals = []         # intervalles entre trames (s)

    def bad_line(self):
        self.n_bad_lines += 1

    def update(self, frame):
        ts = frame.host_ts
        seq = frame.seq
        if self.first_ts is None:
            self.first_ts = ts
        if self.last_ts is not None:
            self.intervals.append(ts - self.last_ts)
        self.last_ts = ts
        self.n_frames += 1
        self._window.append(ts)
        while self._window and ts - self._window[0] > 1.0:
            self._window.pop(0)

        if isinstance(seq, int) and self.last_seq is not None:
            gap = seq - self.last_seq
            if gap == 0:
                self.n_dup += 1
            elif gap < 0:
                self.n_reordered += 1       # ou redemarrage de l'emetteur
            elif gap > 1:
                self.n_lost += gap - 1
        if isinstance(seq, int) and (self.last_seq is None or seq > self.last_seq):
            self.last_seq = seq

    @property
    def fps(self):
        return float(len(self._window))

    @property
    def loss_pct(self):
        expected = self.n_frames + self.n_lost
        return 100.0 * self.n_lost / expected if expected else 0.0

    def summary(self):
        duration = (self.last_ts - self.first_ts) if self.n_frames > 1 else 0.0
        iv = np.array(self.intervals) if self.intervals else np.array([0.0])
        return {
            "frames": self.n_frames,
            "duration_s": round(duration, 2),
            "mean_fps": round(self.n_frames / duration, 1) if duration > 0 else 0.0,
            "lost": self.n_lost,
            "loss_pct": round(self.loss_pct, 2),
            "duplicates": self.n_dup,
            "reordered": self.n_reordered,
            "bad_lines": self.n_bad_lines,
            "interval_mean_ms": round(1000 * float(iv.mean()), 2),
            "interval_p95_ms": round(1000 * float(np.percentile(iv, 95)), 2),
            "interval_max_ms": round(1000 * float(iv.max()), 2),
        }


def subcarrier_quality(amplitudes, zero_tol=1e-6, cv_max=0.5):
    """Carte de qualite par sous-porteuse a partir d'une matrice (temps x sous-porteuses).

    Une sous-porteuse est jugee INVALIDE si elle est nulle la plupart du temps
    (porteuses de garde / DC) ou trop instable (coefficient de variation > cv_max
    mesure dans une piece VIDE, donc sans mouvement)."""
    A = np.asarray(amplitudes, dtype=float)
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    zero_frac = (A <= zero_tol).mean(axis=0)
    cv = np.where(mean > zero_tol, std / np.maximum(mean, zero_tol), np.inf)
    valid = (zero_frac < 0.5) & (cv < cv_max)
    return {"mean": mean, "std": std, "zero_frac": zero_frac, "cv": cv, "valid": valid}


def motion_index(amplitudes, valid, fs, win_s=1.0):
    """Indicateur de mouvement simple : ecart-type temporel de l'amplitude sur une
    fenetre glissante, moyenne sur les sous-porteuses valides, apres normalisation
    de chaque sous-porteuse par sa moyenne. Piece vide -> faible ; mouvement -> fort."""
    A = np.asarray(amplitudes, dtype=float)[:, valid]
    A = A / np.maximum(A.mean(axis=0, keepdims=True), 1e-6)
    w = max(2, int(round(win_s * fs)))
    n = A.shape[0] - w + 1
    if n <= 0:
        return np.array([])
    out = np.empty(n)
    for i in range(n):
        out[i] = A[i:i + w].std(axis=0).mean()
    return out


def correct_ht_scale_jumps(amp_all, n_lltf=64):
    """Corrige un artefact du recepteur ESP32-S3 observe sur nos mesures (2026-10-14) :
    d'une trame a l'autre, toute la partie HT-LTF est parfois multipliee ou divisee par 2
    exactement (saut de 6 dB), alors que le LLTF de la meme trame est stable et que le RSSI
    ne bouge pas. Physiquement, le rapport HT/LLTF d'une meme trame est constant : on le
    ramene a sa mediane par une puissance de 2.

    amp_all : matrice (trames x sous-porteuses) des amplitudes, LLTF (n_lltf) puis HT-LTF.
    Retourne (matrice corrigee, fraction de trames corrigees)."""
    A = np.array(amp_all, dtype=float)
    if A.shape[1] <= n_lltf:
        return A, 0.0
    L, H = A[:, :n_lltf], A[:, n_lltf:]
    lm = np.array([r[r > 0].mean() if (r > 0).any() else np.nan for r in L])
    hm = np.array([r[r > 0].mean() if (r > 0).any() else np.nan for r in H])
    ratio = hm / lm
    ok = np.isfinite(ratio) & (ratio > 0)
    k = np.zeros(A.shape[0])
    k[ok] = np.round(np.log2(ratio[ok] / np.median(ratio[ok])))
    A[:, n_lltf:] = H / (2.0 ** k)[:, None]
    return A, float((k != 0).mean())
