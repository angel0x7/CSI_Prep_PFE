# -*- coding: utf-8 -*-
"""
Étiquettes faibles pour les VRAIES prises (pas de caméra -> pas de keypoints mesurés).

On ne connaît que l'activité de la prise. On en déduit, bin par bin (cfg.FS Hz), le groupe
d'activité (debout / assis / marche / chute / au_sol) :
  - debout, assis, marche : toute la prise ;
  - chute : l'instant de la chute t_f est estimé par détection de rupture sur la forme
    du canal des 4 RX (la position du corps change -> la forme statique change),
    contraint à [0.5 s, 5 s] (protocole : chute "vers 2 s").
    Avant t_f - 0.4 s : debout ; [t_f - 0.4, t_f + 0.8] : chute ; après : au_sol.
"""
import os

import numpy as np
import pandas as pd

from . import config as cfg
from .features import load_take, take_features

REAL_ROOT = os.path.join(os.path.dirname(__file__), "..", "acquisition", "data", "DATASET")
G = {g: i for i, g in enumerate(cfg.GROUPS)}
FALL_BEFORE, FALL_AFTER = 0.4, 0.8


def channel_shape(X):
    """Reconstruit la log-forme du canal (T, 4*26) et le masque (T, 4)."""
    nb, fp = cfg.N_BANDS, cfg.FEAT_PER_RX
    S, M = [], []
    for k in range(len(cfg.RX_NAMES)):
        F = X[:, k * fp:(k + 1) * fp]
        S.append(F[:, :nb] / 10.0 + F[:, nb:2 * nb])
        M.append(F[:, -1])
    return np.concatenate(S, 1), np.stack(M, 1)


def motion_energy(X):
    """Énergie de mouvement (T,) : rms des features 'dyn' sur les 4 RX."""
    nb, fp = cfg.N_BANDS, cfg.FEAT_PER_RX
    return np.sqrt(np.mean([(X[:, k * fp:k * fp + nb] ** 2).mean(1) for k in range(len(cfg.RX_NAMES))], 0))


def fall_time(X, fs=cfg.FS, lo=0.5, hi=5.0, w=1.0):
    """Instant de chute (s) : rupture de forme du canal x pic d'énergie de mouvement."""
    S, M = channel_shape(X)
    W = int(w * fs)
    E = motion_energy(X)
    Es = np.convolve(E, np.ones(int(0.8 * fs)) / int(0.8 * fs), mode="same")
    Es = Es / (np.median(Es) + 1e-6)
    best, best_t, scores = -1, None, []
    for i in range(int(lo * fs), int(hi * fs)):
        a, b = S[max(0, i - W):i], S[i + int(0.6 * fs):i + int(0.6 * fs) + W]
        if len(a) < W // 2 or len(b) < W // 2:
            scores.append(0)
            continue
        d = np.sqrt(((a.mean(0) - b.mean(0)) ** 2).mean())
        noise = np.sqrt(0.5 * (a.var(0).mean() + b.var(0).mean()) / W) + 1e-6
        sc = d / noise * Es[min(i + int(0.3 * fs), len(Es) - 1)]
        scores.append(sc)
        if sc > best:
            best, best_t = sc, i / fs
    return best_t, best, np.array(scores)


def frame_groups(activity, T, t_fall=None, fs=cfg.FS, ignore=0.4):
    """Groupe par bin ; -1 = ignoré dans la perte (zones d'incertitude autour des transitions
    de la chute, dont l'instant n'est qu'estimé : erreur médiane ~0.3 s sur simulation)."""
    y = np.full(T, G.get(activity, -1), dtype=np.int64)
    if activity == "chute":
        t = np.arange(T) / fs
        a, b = t_fall - FALL_BEFORE, t_fall + FALL_AFTER
        y[:] = G["debout"]
        y[(t >= a) & (t < b)] = G["chute"]
        y[t >= b] = G["au_sol"]
        y[(np.abs(t - a) < ignore) | (np.abs(t - b) < ignore)] = -1
    return y


def real_takes(root=REAL_ROOT, persons=(1,), drop_unusable=True):
    idx = pd.read_csv(os.path.join(root, "index.csv"))
    idx = idx[idx.personnes.isin(persons)]
    if drop_unusable:
        idx = idx[idx.qualite != "inutilisable"]
    return idx.reset_index(drop=True)


def build_real(root=REAL_ROOT, persons=(1,), verbose=False):
    """-> liste de dicts {X, y, take, activity, t_fall}."""
    out = []
    for _, r in real_takes(root, persons).iterrows():
        X, t, _ = take_features(load_take(os.path.join(root, r.chemin)), duration=cfg.TAKE_S)
        tf = None
        if r.activite == "chute":
            tf, sc, _ = fall_time(X)
        y = frame_groups(r.activite, len(X), tf)
        out.append(dict(X=X, y=y, take=r.chemin, activity=r.activite, t_fall=tf))
        if verbose and tf is not None:
            print("%-22s t_chute=%.2fs score=%.1f" % (r.chemin, tf, sc))
    return out


if __name__ == "__main__":
    data = build_real(verbose=True)
    tf = np.array([d["t_fall"] for d in data if d["t_fall"] is not None])
    print("chutes : n=%d  t_f médiane=%.2fs  q10=%.2f q90=%.2f" % (len(tf), np.median(tf), *np.percentile(tf, [10, 90])))
