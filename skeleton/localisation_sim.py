# -*- coding: utf-8 -*-
"""
Étude de faisabilité de la LOCALISATION (simulation) : quel montage et quelle précision espérer ?

Méthode testée : empreintes (« fingerprinting ») dans une pièce donnée :
  calibration = pièce vide + personne à des positions connues -> classifieur de zone / régression (x, y).
Montages comparés :
  A  actuel : émetteur S au centre, 4 récepteurs dans les coins (4 liens en étoile)
  B  round-robin : 5 cartes (4 coins + centre) émettent chacune à leur tour, les autres écoutent
     (20 liens qui se croisent ; chaque lien n'a que 1/5 des trames)
Test : MÊME pièce, nouvelles positions, autre personne (taille, réflectivité) ; puis autre pièce
(mobilier / matériel différents) pour montrer que la calibration est propre à chaque pièce.

    python -m skeleton.localisation_sim [-n 300]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config as cfg
from . import synth

DUR = 4.0
CORNERS = {k: np.array(v) for k, v in cfg.RX_POS.items()}
CENTER = np.array(cfg.TX_POS)
LAYOUTS = {
    "A_etoile_4_liens": [("S", n) for n in cfg.RX_NAMES],
    "B_round_robin_20_liens": [(a, b) for a in ["S"] + cfg.RX_NAMES for b in ["S"] + cfg.RX_NAMES if a != b],
}
NODES = dict(CORNERS, S=CENTER)
MARGIN = 0.7


def zone(p, nx, ny):
    fx = (p[0] - MARGIN) / (cfg.ROOM_L - 2 * MARGIN)
    fy = (p[1] - MARGIN) / (cfg.ROOM_W - 2 * MARGIN)
    return int(min(nx - 1, max(0, fx * nx))) + nx * int(min(ny - 1, max(0, fy * ny)))


def link_features(env, links, rng, track, rate_div):
    """Pour chaque lien : forme log-amplitude moyenne (26 bandes, normalisée AGC) + énergie de mouvement."""
    feats = {}
    for tx_name, rx_name in links:
        env.tx = NODES[tx_name]
        env.rx = {rx_name: NODES[rx_name]}
        key = (tx_name, rx_name)
        if key not in env._stat_cache:
            env._stat_cache[key] = env._static(rx_name)
        env.static = {rx_name: env._stat_cache[key]}
        if rx_name not in env.ripple:
            env.ripple[rx_name] = env.ripple[cfg.RX_NAMES[0]]
        act = "vide" if track is None else "debout"
        rx_out, _, _ = synth.simulate_take(act, rng, env, duration=DUR, track=track)
        ts, raw, ids = rx_out[rx_name]
        keep = ids % rate_div == (sum(map(ord, tx_name + rx_name)) % rate_div)   # round-robin : 1 trame sur rate_div
        raw = raw[keep].astype(np.float32)
        H = (raw[:, 1::2] + 1j * raw[:, 0::2])[:, cfg.SC_COLS]
        A = np.abs(H)
        ok = A.mean(1) > 0.5
        A = A[ok] / A[ok].mean(1, keepdims=True)
        A = A.reshape(len(A), cfg.N_BANDS, cfg.SC_POOL).mean(2)
        L = np.log(A + 1e-3)
        feats[key] = np.concatenate([L.mean(0), [L.std(0).mean()]]) if len(L) > 5 else np.zeros(cfg.N_BANDS + 1)
    return feats


def _job(args):
    env_seed, sample_seed, layout, empty = args
    rng = np.random.default_rng(sample_seed)
    env = synth.Environment(env_seed)
    env._stat_cache = {}
    links = LAYOUTS[layout]
    div = len(["S"] + cfg.RX_NAMES) if layout.startswith("B") else 1
    if empty:
        f = link_features(env, links, rng, None, div)
        return None, np.concatenate([f[k] for k in links])
    n = int(DUR * cfg.TX_RATE_HZ)
    kp, atom, pos, heading, info = synth.person_track(rng.choice(["debout", "assis"]), rng, n, 1 / cfg.TX_RATE_HZ)
    p = np.array([rng.uniform(MARGIN, cfg.ROOM_L - MARGIN), rng.uniform(MARGIN, cfg.ROOM_W - MARGIN)])
    pos = pos - pos[0] + p
    f = link_features(env, links, rng, (kp, atom, pos, heading, info), div)
    return p, np.concatenate([f[k] for k in links])


def run(layout, n, env_seeds, workers=None):
    jobs = []
    rng = np.random.default_rng(1)
    for e in env_seeds:
        jobs += [(e, int(rng.integers(2 ** 31)), layout, True) for _ in range(6)]
        jobs += [(e, int(rng.integers(2 ** 31)), layout, False) for _ in range(n)]
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        res = list(ex.map(_job, jobs, chunksize=4))
    out = {}
    k = 0
    for e in env_seeds:
        empty = np.stack([res[k + i][1] for i in range(6)])
        k += 6
        P = np.stack([res[k + i][0] for i in range(n)])
        X = np.stack([res[k + i][1] for i in range(n)])
        k += n
        out[e] = (empty.mean(0), P, X)
    return out


def evaluate(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    envs = list(data)
    res = {}
    # même pièce : 2/3 des positions pour la calibration, 1/3 pour le test
    for nx, ny, name in ((2, 2, "zones_2x2"), (3, 3, "zones_3x3")):
        accs, cross = [], []
        for e in envs:
            base, P, X = data[e]
            Xd = X - base                                   # écart à la pièce vide (calibration)
            y = np.array([zone(p, nx, ny) for p in P])
            cut = int(len(P) * 2 / 3)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
            clf.fit(Xd[:cut], y[:cut])
            accs.append(clf.score(Xd[cut:], y[cut:]))
            o = envs[(envs.index(e) + 1) % len(envs)]      # autre pièce, sans recalibrer le classifieur
            b2, P2, X2 = data[o]
            cross.append(clf.score(X2 - b2, np.array([zone(p, nx, ny) for p in P2])))
        res[name] = {"meme_piece": round(float(np.mean(accs)), 3), "autre_piece": round(float(np.mean(cross)), 3),
                     "hasard": round(1 / (nx * ny), 3)}
    errs = []
    for e in envs:
        base, P, X = data[e]
        Xd = X - base
        cut = int(len(P) * 2 / 3)
        reg = make_pipeline(StandardScaler(), KNeighborsRegressor(7, weights="distance"))
        reg.fit(Xd[:cut], P[:cut])
        errs.append(np.linalg.norm(reg.predict(Xd[cut:]) - P[cut:], axis=1))
    errs = np.concatenate(errs)
    centre = np.array([cfg.ROOM_L / 2, cfg.ROOM_W / 2])
    base_err = np.concatenate([np.linalg.norm(data[e][1] - centre, axis=1) for e in envs])
    res["position_erreur_m"] = {"mediane": round(float(np.median(errs)), 2), "p80": round(float(np.percentile(errs, 80)), 2),
                                "reference_toujours_au_centre": round(float(np.median(base_err)), 2)}
    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300, help="positions par pièce")
    a = ap.parse_args()
    env_seeds = [70_001, 70_002, 70_003]
    out = {}
    for layout in LAYOUTS:
        data = run(layout, a.n, env_seeds)
        out[layout] = evaluate(data)
        print(layout, json.dumps(out[layout], ensure_ascii=False), flush=True)
    path = os.path.join(os.path.dirname(__file__), "results", "localisation_sim.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("->", path)


if __name__ == "__main__":
    main()
