# -*- coding: utf-8 -*-
"""
Inférence du squelette 17 points à partir du CSI.

Hors ligne (une prise du DATASET, ou tout dossier contenant RX1.csv..RX4.csv) :
    python -m skeleton.infer acquisition/data/DATASET/chute/p1/prise_010 [-o sortie.json] [--png]

Temps réel (dashboard) :
    est = SkeletonEstimator()                    # charge skeleton/checkpoints/skeleton_tcn.pt
    est.push("RX1", t_ms, raw128)                # à chaque trame UDP reçue (valeurs v0..v127)
    frame = est.latest(x=4.2, y=3.1, heading=0)  # dict JSON-sérialisable (position fournie par la localisation)

Format d'une frame :
    t, activity, confidence, probs{groupe: p}, height_m,
    keypoints_body  [[x,y,z]*17]  repère corps (m) : x gauche, y avant, z haut
    keypoints_world [[x,y,z]*17]  repère salle (m), si x/y/heading fournis
    keypoints_coco  [[u,v,vis]*17] projection de face, image 192x256 (format COCO)
    present (bool)
"""
import argparse
import collections
import json
import os

import numpy as np
import torch

from . import config as cfg
from .features import load_take, raw_to_complex, take_features
from .labels import motion_energy
from .model import SkeletonTCN
from .poses import KEYPOINTS, SKELETON, pose_fall, pose_walk, to_coco_2d, to_world

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_CKPT = os.path.join(CKPT_DIR, "skeleton_tcn.pt")        # temps réel (causal)
CLIP_CKPT = os.path.join(CKPT_DIR, "skeleton_clip.pt")          # prise entière (8 s, non causal)
STANDARD_GAIT_HZ = 0.9          # cycles de marche / s (≈ 1.8 pas/s, marche normale)
G = {g: i for i, g in enumerate(cfg.GROUPS)}


def load_model(path=DEFAULT_CKPT):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    m = SkeletonTCN(causal=c.get("causal", True), dilations=tuple(c.get("dilations", (1, 2, 4, 8, 16))))
    m.load_state_dict(ck["state_dict"])
    m.channel_mask = ck["channel_mask"]
    m.warmup_s = (m.receptive_field - 1) / cfg.FS if m.causal else 0.0
    m.mode = "temps_reel" if m.causal else "prise_entiere"
    m.eval()
    return m


# --------------------------------------------------------------------------
# Animation : squelette qui bouge sur toute la prise
# --------------------------------------------------------------------------
def _smooth(x, k=5):
    """Moyenne glissante centrée le long du temps (bords répliqués)."""
    if k <= 1:
        return x
    pad = np.pad(x, [(k // 2, k // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros_like(pad[:1]), pad]), 0)
    return (c[k:] - c[:-k]) / k


def gait_cadence(E, fs=cfg.FS, min_contrast=3.0):
    """Cadence de marche (cycles/s) mesurée sur l'énergie de mouvement CSI.
    Utilisée seulement si le pic est net ; sinon cadence standard (cf. README : sur nos
    prises, la cadence ne ressort pas du bruit -> cadence standard la plupart du temps)."""
    if len(E) >= 40:
        e = (E - E.mean()) * np.hanning(len(E))
        F = np.abs(np.fft.rfft(e))
        f = np.fft.rfftfreq(len(E), 1 / fs)
        band = (f >= 1.0) & (f <= 2.6)                       # fréquence des PAS
        if band.any():
            k = np.argmax(F * band)
            contrast = F[k] / (np.median(F[band]) + 1e-9)
            if contrast >= min_contrast:
                return f[k] / 2, "mesurée sur le CSI (contraste %.1f)" % contrast
    return STANDARD_GAIT_HZ, "standard (non mesurable sur ce signal)"


def animate(out, X, fs=cfg.FS, ema=0.45):
    """Squelette d'affichage par image, animé sur toute la prise :
      - activité lissée (évite le clignotement) ;
      - marche : cycle de foulée continu (cadence mesurée si possible, sinon standard) ;
      - chute : avancement monotone 0 -> 1 estimé par le modèle (atomes de chute) ;
      - autres : squelette du modèle ;
      - lissage temporel aller-retour (sans retard) des keypoints.
    Retourne (kp (T,17,3), groupes (T,), infos)."""
    T = len(X)
    P = _smooth(np.exp(out["group_logp"]), 5)
    g = P.argmax(1)
    W = np.exp(out["atom_logits"] - out["atom_logits"].max(-1, keepdims=True))
    W /= W.sum(-1, keepdims=True)
    H = float(np.median(out["height"]))
    walking = g == G["marche"]
    E = motion_energy(X)
    gait_hz, gait_src = gait_cadence(E[walking] if walking.sum() >= 40 else E)
    phase = 2 * np.pi * gait_hz * np.arange(T) / fs
    # avancement de la chute : sur chaque segment 'chute' continu, progression 0 -> 1
    # (debout -> au sol), mélangée à l'estimation du modèle (atomes de chute) mais forcée monotone
    nw, nf = cfg.N_WALK_ATOMS, cfg.N_FALL_ATOMS
    fa = W[:, 3 + nw:3 + nw + nf]
    a_model = (fa * ((np.arange(nf) + 0.5) / nf)).sum(1) / np.maximum(fa.sum(1), 1e-6)
    alpha = np.zeros(T)
    t0 = None
    for t in range(T + 1):
        inside = t < T and g[t] == G["chute"]
        if inside and t0 is None:
            t0 = t
        if not inside and t0 is not None:
            n = t - t0
            ramp = np.linspace(0.05, 0.95, n) if n > 1 else np.array([0.5])
            alpha[t0:t] = np.maximum.accumulate(0.6 * ramp + 0.4 * a_model[t0:t])
            t0 = None
    kp = out["kp"].astype(float).copy()
    for t in range(T):
        if g[t] == G["vide"]:
            kp[t] = 0
        elif g[t] == G["marche"]:
            kp[t] = pose_walk(phase[t] % (2 * np.pi), H, stride=min(1.0, 0.4 + P[t, G["marche"]]))
        elif g[t] == G["chute"]:
            kp[t] = pose_fall(alpha[t], H)
    # lissage aller-retour (zéro retard) sauf aux apparitions / disparitions
    for direction in (1, -1):
        rng_t = range(1, T) if direction == 1 else range(T - 2, -1, -1)
        for t in rng_t:
            prev = kp[t - direction]
            if np.any(prev) and np.any(kp[t]):
                kp[t] = ema * kp[t] + (1 - ema) * prev
    return kp, g, {"gait_hz": round(float(gait_hz), 2), "gait_source": gait_src, "height_m": round(H, 2)}


@torch.no_grad()
def run(model, X):
    out = model(torch.from_numpy(X[None].astype(np.float32)) * model.channel_mask)
    return {k: v[0].numpy() for k, v in out.items()}


def make_frame(out, i, t, x=None, y=None, heading=None, kp=None, g=None):
    probs = np.exp(out["group_logp"][i])
    g = int(probs.argmax()) if g is None else int(g)
    raw = out["kp"][i].astype(float)
    kp = raw if kp is None else kp
    fr = {"t": round(float(t), 3), "activity": cfg.GROUPS[g], "confidence": round(float(probs[g]), 3),
          "probs": {k: round(float(p), 3) for k, p in zip(cfg.GROUPS, probs)},
          "present": bool(probs[0] < 0.5), "height_m": round(float(out["height"][i]), 3),
          "keypoints_body": np.round(kp, 3).tolist(),
          "keypoints_coco": np.round(to_coco_2d(kp), 1).tolist(),
          "keypoints_model": np.round(raw, 3).tolist()}
    if x is not None and y is not None:
        fr["keypoints_world"] = np.round(to_world(kp, x, y, heading or 0.0), 3).tolist()
    return fr


def infer_take(take_dir, model=None, x=None, y=None, heading=None, animated=True):
    """Analyse une prise entière (8 s) : une image toutes les 50 ms.
    keypoints_body = squelette animé (affichage) ; keypoints_model = sortie brute du réseau."""
    model = model or (load_model(CLIP_CKPT) if os.path.exists(CLIP_CKPT) else load_model())
    rx = load_take(take_dir)
    X, t, info = take_features(rx)
    out = run(model, X)
    if animated:
        kp, g, anim = animate(out, X)
    else:
        kp, g, anim = [None] * len(t), [None] * len(t), {}
    frames = [make_frame(out, i, t[i], x, y, heading, kp[i], g[i]) for i in range(len(t))]
    return {"source": take_dir.replace("\\", "/"), "fs": cfg.FS, "keypoint_names": KEYPOINTS,
            "skeleton": SKELETON, "mode": getattr(model, "mode", "temps_reel"),
            "warmup_s": getattr(model, "warmup_s", 0.0), "animation": anim,
            "rx_frames": info["frames"], "frames": frames}


class SkeletonEstimator(object):
    """Estimation en flux : garde les dernières `buffer_s` secondes de chaque RX."""

    def __init__(self, ckpt=DEFAULT_CKPT, buffer_s=6.0):
        self.model = load_model(ckpt)
        self.buffer_s = buffer_s
        self.buf = {n: collections.deque() for n in cfg.RX_NAMES}

    def push(self, rx_name, t_ms, raw128):
        q = self.buf[rx_name]
        q.append((int(t_ms), np.asarray(raw128, dtype=np.int16)))
        while q and q[-1][0] - q[0][0] > self.buffer_s * 1000:
            q.popleft()

    def latest(self, x=None, y=None, heading=None):
        rx = {}
        for n, q in self.buf.items():
            if len(q) >= 5:
                t = np.array([a for a, _ in q], np.int64)
                rx[n] = (t, raw_to_complex(np.stack([b for _, b in q])))
        if not rx:
            return None
        t_end = max(int(v[0][-1]) for v in rx.values())
        t_start = t_end - int(self.buffer_s * 1000)
        X, t, _ = take_features(rx, t_start_ms=t_start, duration=self.buffer_s)
        out = run(self.model, X)
        return make_frame(out, len(t) - 1, t_end / 1000.0, x, y, heading)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--png", action="store_true", help="planche de squelettes (matplotlib)")
    a = ap.parse_args()
    res = infer_take(a.take_dir, load_model(a.ckpt))
    out = a.out or os.path.join(a.take_dir, "skeleton_pred.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    acts = collections.Counter(fr["activity"] for fr in res["frames"])
    print(out, dict(acts))
    if a.png:
        from .visualize import plot_prediction
        print(plot_prediction(out))


if __name__ == "__main__":
    main()
