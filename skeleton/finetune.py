# -*- coding: utf-8 -*-
"""
Affiner le modèle temps réel sur un NOUVEL environnement à partir de sessions guidées (session.py).

    python -m skeleton.finetune acquisition/data/MAISON                 # toutes les sessions du dossier
    python -m skeleton.finetune acquisition/data/MAISON --nom maison --steps 800

Étapes :
  1. chaque session -> features 20 Hz + étiquettes déduites des consignes
     (REACTION_S ignoré après chaque consigne ; dans 'chute', instant de chute estimé
     -> 'chute' autour, 'au_sol' après) ; découpe en séquences de 16 s ;
  2. si >= 2 sessions : évaluation honnête AVANT / APRÈS, en laissant chaque session de côté
     tour à tour (affinage sans elle, test en direct rejoué sur elle) ;
  3. affinage final sur toutes les sessions -> skeleton/checkpoints/skeleton_tcn_<nom>.pt
     (on repart du modèle actuel et on continue de lui montrer les anciennes données,
     pour qu'il n'oublie pas la salle de classe).
Ensuite : python -m skeleton.session ... --ckpt skeleton/checkpoints/skeleton_tcn_<nom>.pt
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import torch

from . import config as cfg
from . import labels as L
from .dataset import SEQ_T, load
from .features import load_take, take_features
from .infer import DEFAULT_CKPT, load_model
from .live import LiveClassifier, replay
from .session import REACTION_S, expected_ok
from .train import synth_split, train_model

G = {g: i for i, g in enumerate(cfg.GROUPS)}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_session(d):
    with open(os.path.join(d, "consignes.json"), encoding="utf-8") as f:
        meta = json.load(f)
    rx = load_take(d)
    dur = meta["consignes"][-1][1]
    X, _, _ = take_features(rx, t_start_ms=meta["t0_ms"], duration=dur)
    T = len(X)
    t = (np.arange(T) + 0.5) / cfg.FS
    y = np.full(T, -1, np.int64)
    for a, b, s in meta["consignes"]:
        m = (t >= a + REACTION_S) & (t < b)
        if s != "chute":
            y[m] = G[s]
            continue
        # chute : instant estimé dans le segment (rupture du canal x pic de mouvement)
        i0, i1 = int(a * cfg.FS), int(b * cfg.FS)
        tf, _, _ = L.fall_time(X[i0:i1], lo=0.3, hi=min(4.0, (b - a) - 1.0))
        tf = a + (tf if tf is not None else 1.5)
        y[(t >= tf - L.FALL_BEFORE + 0.2) & (t < tf + L.FALL_AFTER - 0.2)] = G["chute"]
        y[(t >= tf + L.FALL_AFTER + 0.4) & (t < b)] = G["au_sol"]
    return {"dir": d, "meta": meta, "X": X.astype(np.float16), "y": y, "rx": rx}


def chunks(sessions, stride_s=4.0):
    """Sessions -> séquences de 16 s (format du cache 'seq')."""
    Xs, ys = [], []
    for s in sessions:
        X, y = s["X"], s["y"]
        if len(X) < SEQ_T:
            pad = SEQ_T - len(X)
            X = np.concatenate([X, np.repeat(X[-1:], pad, 0)])
            y = np.concatenate([y, np.full(pad, -1)])
        for i in range(0, len(X) - SEQ_T + 1, int(stride_s * cfg.FS)):
            Xs.append(X[i:i + SEQ_T])
            ys.append(y[i:i + SEQ_T])
    return {"X": np.stack(Xs), "y": np.stack(ys)}


def evaluate(model, sess):
    """Rejoue la session en direct (même moteur que session.py) et compare aux consignes."""
    raw = {}
    for n in cfg.RX_NAMES:
        p = os.path.join(sess["dir"], n + ".csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            a = np.loadtxt(p, delimiter=",", dtype=np.int64, ndmin=2)
            raw[n] = (a[:, 0], a[:, 1:129].astype(np.int16))
    segs = sess["meta"]["consignes"]
    seq = {"rx": raw, "t0_ms": sess["meta"]["t0_ms"], "duration": segs[-1][1]}
    preds = list(replay(seq, LiveClassifier(model), realtime=False))
    ok = n = 0
    per = collections.defaultdict(lambda: [0, 0])
    for p in preds:
        seg = next((s for s in segs if s[0] <= p["t"] < s[1]), None)
        if p["etat"] == "analyse" or seg is None or p["t"] - seg[0] < REACTION_S:
            continue
        g = expected_ok(p["etat"], seg[2])
        ok, n = ok + g, n + 1
        per[seg[2]][0] += g
        per[seg[2]][1] += 1
    fall = next((s for s in segs if s[2] == "chute"), None)
    alerts = [p["t"] for p in preds if p["evenement"] and p["evenement"]["type"] == "alerte_chute"]
    return {"accord": round(ok / max(n, 1), 3),
            "par_etat": {k: round(v[0] / max(v[1], 1), 2) for k, v in per.items()},
            "chute_alertee": bool(fall and any(t >= fall[0] - 0.5 for t in alerts)) if fall else None,
            "fausses_alertes": sum(1 for t in alerts if fall is None or t < fall[0] - 0.5)}


def finetune(base_ckpt, sessions, steps, lr):
    syn, real = load("synth"), load("real_aug")
    tr_s, _ = synth_split(syn)
    base = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    m = train_model(syn, real, tr_s, np.arange(len(real["X"])), steps=steps, seq=chunks(sessions),
                    init=base["state_dict"], lr=lr, verbose=False)
    return m, base


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dossier", help="dossier contenant des session_* (ou une seule session)")
    ap.add_argument("--nom", default="maison")
    ap.add_argument("--base", default=DEFAULT_CKPT)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dirs = sorted(glob.glob(os.path.join(a.dossier, "session_*"))) or [a.dossier]
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "consignes.json"))]
    if not dirs:
        raise SystemExit("aucune session (consignes.json) dans %s" % a.dossier)
    sessions = [load_session(d) for d in dirs]
    lab = sum(int((s["y"] >= 0).sum()) for s in sessions)
    print("%d session(s), %.1f min, %d bins étiquetés" % (len(sessions), sum(len(s["X"]) for s in sessions) / cfg.FS / 60, lab))
    base_model = load_model(a.base)
    report = {"sessions": [os.path.basename(d) for d in dirs], "avant_apres": []}
    if len(sessions) >= 2:
        print("Évaluation laisser-une-session-de-côté (avant -> après affinage) :")
        for k, s in enumerate(sessions):
            m, _ = finetune(a.base, [x for j, x in enumerate(sessions) if j != k], a.steps, a.lr)
            before, after = evaluate(base_model, s), evaluate(m, s)
            report["avant_apres"].append({"session": os.path.basename(s["dir"]), "avant": before, "apres": after})
            print("  %s : accord %.0f%% -> %.0f%% | fausses alertes %d -> %d | chute alertée %s -> %s" % (
                os.path.basename(s["dir"]), 100 * before["accord"], 100 * after["accord"],
                before["fausses_alertes"], after["fausses_alertes"], before["chute_alertee"], after["chute_alertee"]))
    else:
        print("Une seule session : pas d'évaluation honnête possible (il en faut au moins 2).")
        report["avant"] = evaluate(base_model, sessions[0])
        print("  modèle actuel sur cette session :", report["avant"])
    m, base = finetune(a.base, sessions, a.steps, a.lr)
    out = os.path.join(HERE, "checkpoints", "skeleton_tcn_%s.pt" % a.nom)
    torch.save({"state_dict": m.state_dict(), "channel_mask": m.channel_mask,
                "config": dict(base.get("config", {}), affine_sur=report["sessions"], base=os.path.basename(a.base))}, out)
    with open(os.path.join(HERE, "results", "finetune_%s.json" % a.nom), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("Modèle affiné ->", out)


if __name__ == "__main__":
    main()
