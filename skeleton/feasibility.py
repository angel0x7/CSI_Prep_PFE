# -*- coding: utf-8 -*-
"""
Étude de faisabilité : le CSI de nos 4 ESP32 contient-il de l'information de POSTURE
au-delà de l'activité ?

Sur des séances SIMULÉES jamais vues (vérité terrain squelette parfaite), on compare l'erreur
du modèle (MPJPE, mm) à des références qui n'utilisent PAS le CSI pour la posture :
  B0  toujours le squelette 'debout' moyen
  B1  activité seule : un squelette type par activité (vraie activité donnée)
  B2  oracle du décodeur : le bon atome (phase du pas / avancement de chute exacts) x la vraie taille
Si le modèle ≈ B1, le CSI ne lui apprend rien sur la posture au-delà de l'activité.

    python -m skeleton.feasibility [--ckpt skeleton/checkpoints/skeleton_clip.pt]
"""
import argparse
import json
import os

import numpy as np

from . import config as cfg
from .dataset import load
from .infer import CLIP_CKPT, load_model
from .poses import atom_bank
from .train import predict, synth_split, train_model

KP10 = [0, 5, 6, 9, 10, 11, 12, 13, 14, 15]   # nez, épaules, poignets, hanches, genoux, cheville G
G = {g: i for i, g in enumerate(cfg.GROUPS)}


def mpjpe(a, b, idx=None):
    if idx is not None:
        a, b = a[..., idx, :], b[..., idx, :]
    return np.linalg.norm(a - b, axis=-1).mean(-1)          # (N,T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CLIP_CKPT)
    ap.add_argument("--synth-cache", default="synth", help="synth | synth_ideal (acquisition parfaite)")
    ap.add_argument("--train-steps", type=int, default=0,
                    help=">0 : entraîne un modèle 'prise entière' sur ce cache (simulé seul) au lieu de --ckpt")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    syn = load(a.synth_cache)
    tr, te = synth_split(syn)
    if a.train_steps:
        real = load("real")
        m = train_model(syn, real, tr, np.arange(len(real["X"])), use_real=False, clip=True,
                        steps=a.train_steps, verbose=True)
    else:
        m = load_model(a.ckpt)
    out = predict(m, syn["X"][te])
    kp = syn["kp"][te].astype(np.float32)
    atom = syn["atom"][te].astype(int)
    grp = syn["group"][te].astype(int)
    H = syn["height"][te][:, None, None, None]
    A = atom_bank()
    nw, nf = cfg.N_WALK_ATOMS, cfg.N_FALL_ATOMS
    tmpl = {G["debout"]: A[1], G["assis"]: A[2], G["marche"]: A[3:3 + nw].mean(0),
            G["chute"]: A[3 + nw:3 + nw + nf].mean(0), G["au_sol"]: A[-1], G["vide"]: A[0]}
    B1 = np.stack([tmpl[g] for g in grp.ravel()]).reshape(kp.shape)
    B2 = A[atom] * (H / cfg.NOMINAL_HEIGHT)
    B0 = np.broadcast_to(A[1], kp.shape)
    pres = grp != G["vide"]
    w = int(getattr(m, "warmup_s", 0) * cfg.FS)
    pres[:, :w] = False                                     # mode temps réel : on retire le démarrage
    res = {}
    for name, idx in (("17 points", None), ("10 points", KP10)):
        rows = {}
        for g in ["debout", "assis", "marche", "chute", "au_sol"]:
            mm = pres & (grp == G[g])
            rows[g] = {k: round(float(mpjpe(v, kp, idx)[mm].mean() * 1000))
                       for k, v in (("B0_debout", B0), ("B1_activite", B1), ("modele_CSI", out["kp"]),
                                    ("B2_oracle", B2))}
        rows["TOUT"] = {k: round(float(mpjpe(v, kp, idx)[pres].mean() * 1000))
                        for k, v in (("B0_debout", B0), ("B1_activite", B1), ("modele_CSI", out["kp"]),
                                     ("B2_oracle", B2))}
        res[name] = rows
    # le modèle suit-il les variations À L'INTÉRIEUR d'une activité ? (taille, phase du pas)
    hp = out["height"][pres]
    ht = np.broadcast_to(syn["height"][te][:, None], grp.shape)[pres]
    res["taille_correlation"] = round(float(np.corrcoef(hp, ht)[0, 1]), 3)
    wk = pres & (grp == G["marche"])
    if wk.any():
        Wp = out["atom_logits"][..., 3:3 + nw]
        ph_pred = np.argmax(Wp, -1)[wk]
        ph_true = (atom[wk] - 3) % nw
        d = np.minimum(np.abs(ph_pred - ph_true), nw - np.abs(ph_pred - ph_true))
        res["phase_du_pas_juste_a_1_atome_pres"] = round(float((d <= 1).mean()), 3)
        res["phase_du_pas_hasard"] = round(3 / nw, 3)
    res["cache"] = a.synth_cache
    print(json.dumps(res, indent=1, ensure_ascii=False))
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    name = "faisabilite.json" if a.synth_cache == "synth" else "faisabilite_%s.json" % a.synth_cache
    with open(os.path.join(os.path.dirname(__file__), "results", name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
