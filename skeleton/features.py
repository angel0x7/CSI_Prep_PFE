# -*- coding: utf-8 -*-
"""
CSV des 4 récepteurs -> matrice de features (T, N_IN) sur une grille commune à cfg.FS Hz.

Même chemin de code pour les vraies prises et les prises simulées (pas de biais de pipeline).

Par récepteur et par bin de 50 ms :
  - amplitude des 52 sous-porteuses utiles, normalisée trame par trame par sa moyenne
    (supprime l'AGC), moyennée par paires -> 26 bandes, en log ;
  - 'dyn'    = log-amplitude - moyenne des 2 s passées (mouvement, ~0 si immobile) ;
  - 'static' = moyenne des 2 s passées (forme statique du canal = où/comment est le corps) ;
  - 'niveau' = log du niveau moyen, relatif à sa moyenne des 4 s passées ;
  - 'masque' = 1 si au moins une trame reçue dans le bin, 0 sinon (trou interpolé).
Tout est CAUSAL (passé uniquement) : mêmes features hors ligne et en temps réel
(seule exception : un trou de quelques bins est interpolé avec le bin reçu suivant ;
en flux, les derniers bins manquants sont maintenus à la dernière valeur).
La phase n'est pas utilisée : sur ESP32 sans référence commune, elle est dominée par
le CFO/STO (rotation aléatoire à chaque trame).
"""
import os

import numpy as np

from . import config as cfg

STATIC_WIN_S = 2.0          # fenêtre (passée) de la forme statique
LEVEL_WIN_S = 4.0           # fenêtre (passée) de référence du niveau

# Règles de nettoyage reprises de csi_preprocess.py (pré-traitement de l'équipe) : mêmes paramètres.
try:
    from csi_preprocess import Config as _PrepConfig
    _P = _PrepConfig()
    MAX_BAD_SC_FRAC, HAMPEL_WIN, HAMPEL_SIGMA, HAMPEL_REL_FLOOR = (
        _P.max_bad_sc_frac, _P.hampel_win, _P.hampel_sigma, _P.hampel_rel_floor)
except ImportError:                                          # module lancé hors de la racine du dépôt
    MAX_BAD_SC_FRAC, HAMPEL_WIN, HAMPEL_SIGMA, HAMPEL_REL_FLOOR = 0.30, 11, 5.0, 0.10
USE_HAMPEL = False          # désactivé par défaut : voir README (gain non démontré, cf. EVAL.md)


def clean_frames(t_ms, H, use_hampel=None):
    """Nettoyage repris de csi_preprocess.py, en version CAUSALE (utilisable en temps réel) :
      1. rejet des trames dont > 30 % des sous-porteuses utiles sont nulles (trame corrompue) ;
      2. (option) Hampel sur l'amplitude de chaque sous-porteuse : médiane + MAD des
         HAMPEL_WIN trames PRÉCÉDENTES (csi_preprocess utilise une fenêtre centrée, qui regarde
         le futur) ; un point aberrant est remplacé par la médiane, la phase est conservée."""
    null = (H.real == 0) & (H.imag == 0)
    ok = null.mean(1) <= MAX_BAD_SC_FRAC
    t_ms, H = t_ms[ok], H[ok]
    if (USE_HAMPEL if use_hampel is None else use_hampel) and len(H) > HAMPEL_WIN:
        A = np.abs(H)
        w = HAMPEL_WIN
        past = np.lib.stride_tricks.sliding_window_view(A[:-1], w, axis=0)   # (n-w, K, w)
        med = np.median(past, -1)
        mad = np.median(np.abs(past - med[..., None]), -1)
        x = A[w:]
        thr = np.maximum(HAMPEL_SIGMA * 1.4826 * mad, HAMPEL_REL_FLOOR * med) + 1e-9
        bad = np.abs(x - med) > thr
        if bad.any():
            H = H.copy()
            Hw = H[w:]
            Hw[bad] = med[bad] * np.exp(1j * np.angle(Hw[bad]))
    return t_ms, H


def load_rx_csv(path):
    """-> (t_ms int64 (n,), H complexe (n, 52)) ; None si vide/absent."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    a = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    if a.shape[0] == 0 or a.shape[1] < 129:
        return None
    return a[:, 0], raw_to_complex(a[:, 1:129])


def raw_to_complex(raw):
    raw = np.asarray(raw, dtype=np.float32)
    H = raw[:, 1::2] + 1j * raw[:, 0::2]                   # [imag, réel] entrelacés
    return H[:, cfg.SC_COLS]


def load_take(take_dir):
    """{RXn: (t_ms, H)} pour une prise (RX manquants absents du dict)."""
    out = {}
    for name in cfg.RX_NAMES:
        r = load_rx_csv(os.path.join(take_dir, name + ".csv"))
        if r is not None and len(r[0]) >= 5:
            out[name] = r
    return out


def _trailing_mean(x, w):
    """Moyenne des w derniers bins (passé uniquement -> causal, identique hors ligne et en flux)."""
    x2 = x if x.ndim == 2 else x[:, None]
    c = np.cumsum(np.pad(x2, ((1, 0), (0, 0))), 0)
    n = len(x2)
    i = np.arange(n)
    lo = np.clip(i + 1 - w, 0, n)
    out = (c[i + 1] - c[lo]) / (i + 1 - lo)[:, None]
    return out if x.ndim == 2 else out[:, 0]


def take_features(rx, t_start_ms=None, duration=None, fs=cfg.FS):
    """rx = {RXn: (t_ms, H)} -> (X (T, N_IN) float32, t (T,) secondes depuis t_start, info).

    Les RX absents donnent des features nulles et un masque à 0.
    """
    if not rx:
        raise ValueError("aucun récepteur")
    if t_start_ms is None:
        t_start_ms = max(int(v[0][0]) for v in rx.values())   # début commun
    if duration is None:
        t_end = min(int(v[0][-1]) for v in rx.values())
        duration = max(1.0 / fs, (t_end - t_start_ms) / 1000.0)
    T = int(round(duration * fs))
    feats = []
    n_frames = {}
    for name in cfg.RX_NAMES:
        F = np.zeros((T, cfg.FEAT_PER_RX), np.float32)
        if name in rx:
            t_ms, H = clean_frames(*rx[name])
            A = np.abs(H)
            lvl = A.mean(1)
            ok = lvl > 0.5
            A, t_ms, lvl = A[ok], t_ms[ok], lvl[ok]
            b = np.floor((t_ms - t_start_ms) / 1000.0 * fs).astype(int)
            sel = (b >= 0) & (b < T)
            b, A, lvl = b[sel], A[sel], lvl[sel]
            n_frames[name] = int(len(b))
            if len(b):
                An = A / lvl[:, None]
                An = An.reshape(len(An), cfg.N_BANDS, cfg.SC_POOL).mean(2)
                cnt = np.bincount(b, minlength=T).astype(np.float32)
                S = np.zeros((T, cfg.N_BANDS), np.float32)
                np.add.at(S, b, np.log(An + 1e-3))
                L = np.bincount(b, weights=np.log(lvl), minlength=T)
                m = cnt > 0
                S[m] /= cnt[m, None]
                L[m] /= cnt[m]
                # interpolation des bins vides
                tb = np.arange(T)
                if m.sum() >= 2:
                    for j in range(cfg.N_BANDS):
                        S[~m, j] = np.interp(tb[~m], tb[m], S[m, j])
                    L[~m] = np.interp(tb[~m], tb[m], L[m])
                else:
                    S[:] = S[m].mean(0) if m.any() else 0
                    L[:] = L[m].mean() if m.any() else 0
                stat = _trailing_mean(S, int(STATIC_WIN_S * fs))
                F[:, :cfg.N_BANDS] = (S - stat) * 10.0          # dyn, ~unité
                F[:, cfg.N_BANDS:2 * cfg.N_BANDS] = stat - stat.mean(1, keepdims=True)
                F[:, -2] = (L - _trailing_mean(L, int(LEVEL_WIN_S * fs))) * 5.0
                F[:, -1] = m
        else:
            n_frames[name] = 0
        feats.append(F)
    X = np.concatenate(feats, 1)
    t = np.arange(T) / fs
    return X, t, {"t_start_ms": int(t_start_ms), "frames": n_frames}


def sim_to_rx(rx_out):
    """Sortie de synth.simulate_take -> même format que load_take."""
    return {n: (ts, raw_to_complex(raw)) for n, (ts, raw, _ids) in rx_out.items()}
