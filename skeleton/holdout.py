# -*- coding: utf-8 -*-
"""
Prises réelles MISES DE CÔTÉ pour les tests en direct, et modèle de test entraîné sans elles.

20 % des prises p1 utilisables de chaque activité (tirage fixe) ne servent jamais à l'entraînement
du modèle 'skeleton_tcn_holdout.pt' (ni elles, ni leurs copies augmentées). Les montages réels
du banc de test (sequences.py, source 'reel') n'utilisent que ces prises : le test est honnête.

    python -m skeleton.holdout          # tire les prises (results/holdout_prises.json) + entraîne le modèle
"""
import argparse
import json
import os

import numpy as np

from . import config as cfg
from .dataset import load
from .train import synth_split, train_model

HERE = os.path.dirname(__file__)
LIST = os.path.join(HERE, "results", "holdout_prises.json")
CKPT = os.path.join(HERE, "checkpoints", "skeleton_tcn_holdout.pt")


def choose(frac=0.2, seed=0):
    from .labels import real_takes
    idx = real_takes()
    rng = np.random.default_rng(seed)
    out = {}
    for act, grp in idx.groupby("activite"):
        takes = sorted(grp.chemin)
        k = max(2, int(round(frac * len(takes))))
        out[act] = sorted(rng.choice(takes, k, replace=False).tolist())
    return out


def holdout_takes():
    if not os.path.exists(LIST):
        raise FileNotFoundError("lancer d'abord : python -m skeleton.holdout")
    with open(LIST, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-seq", action="store_true", help="sans les séquences multi-états")
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    import torch
    ho = choose()
    os.makedirs(os.path.dirname(LIST), exist_ok=True)
    with open(LIST, "w", encoding="utf-8") as f:
        json.dump(ho, f, indent=1, ensure_ascii=False)
    print({k: len(v) for k, v in ho.items()}, "prises mises de côté")
    syn, real = load("synth"), load("real_aug")
    held = {t for v in ho.values() for t in v}
    keep = np.array([t not in held for t in real["take"]])           # 'take' = prise d'origine (aussi pour les copies)
    print("entraînement sur %d prises réelles (dont augmentées), %d exclues" % (keep.sum(), (~keep).sum()))
    tr_s, _ = synth_split(syn)
    seq_path = os.path.join(HERE, "cache", "seq_holdout.npz")
    seq = load("seq_holdout") if os.path.exists(seq_path) and not a.no_seq else None
    print("séquences multi-états :", "oui (%d)" % len(seq["X"]) if seq is not None else "non")
    m = train_model(syn, real, tr_s, np.nonzero(keep)[0], steps=a.steps, verbose=True, seq=seq)
    torch.save({"state_dict": m.state_dict(), "channel_mask": m.channel_mask,
                "config": {"fs": cfg.FS, "groups": cfg.GROUPS, "causal": True,
                           "dilations": list(m.dilations), "holdout": LIST, "sequences": seq is not None}}, CKPT)
    print("->", CKPT)


if __name__ == "__main__":
    main()
