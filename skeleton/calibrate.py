# -*- coding: utf-8 -*-
"""
Calibration simulateur <-> vraies prises : compare, par activité, les statistiques des features
(énergie de mouvement 'dyn', relief statique, variation de niveau, taux de bins reçus).

    python -m skeleton.calibrate                       # stats réelles vs simulées (SIGMA_BODY actuel)
    python -m skeleton.calibrate --sweep 0.3 0.6 1 2   # balayage de la réflectivité du corps
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import config as cfg
from . import synth
from .features import load_take, sim_to_rx, take_features

REAL_ROOT = os.path.join(os.path.dirname(__file__), "..", "acquisition", "data", "DATASET")
NB = cfg.N_BANDS


def stats(X):
    out = {}
    for k in range(len(cfg.RX_NAMES)):
        F = X[:, k * cfg.FEAT_PER_RX:(k + 1) * cfg.FEAT_PER_RX]
        m = F[:, -1] > 0
        if m.sum() < 10:
            continue
        out.setdefault("dyn", []).append(np.sqrt((F[m, :NB] ** 2).mean()))
        out.setdefault("static", []).append(F[m, NB:2 * NB].std())
        out.setdefault("level", []).append(F[m, -2].std())
        out.setdefault("recv", []).append(m.mean())
    return {k: float(np.mean(v)) for k, v in out.items()}


def real_stats():
    idx = pd.read_csv(os.path.join(REAL_ROOT, "index.csv"))
    rows = []
    for _, r in idx[(idx.personnes == 1) & (idx.qualite != "inutilisable")].iterrows():
        X, _, _ = take_features(load_take(os.path.join(REAL_ROOT, r.chemin)), duration=cfg.TAKE_S)
        rows.append(dict(activite=r.activite, **stats(X)))
    return pd.DataFrame(rows)


def sim_stats(sigma, n=12, seed=0):
    rng = np.random.default_rng(seed)
    envs = [synth.Environment(100 + i) for i in range(4)]
    rows = []
    for act in synth.ACTIVITIES:
        for i in range(n):
            rx_out, lab, info = synth.simulate_take(act, rng, envs[i % 4],
                                                    sigma=sigma * rng.uniform(0.6, 1.4))
            X, _, _ = take_features(sim_to_rx(rx_out), t_start_ms=lab["t0_ms"] + 5, duration=cfg.TAKE_S)
            rows.append(dict(activite=act, **stats(X)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=float, nargs="*", default=None)
    ap.add_argument("-n", type=int, default=12)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    r = real_stats().groupby("activite").median()
    print("=== RÉEL (médianes) ===")
    print(r.round(3))
    for s in (a.sweep or [synth.SIGMA_BODY]):
        print("\n=== SIMULÉ sigma=%.3g ===" % s)
        print(sim_stats(s, a.n).groupby("activite").median().round(3))


if __name__ == "__main__":
    main()
