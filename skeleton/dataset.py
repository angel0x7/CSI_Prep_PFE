# -*- coding: utf-8 -*-
"""
Construction des jeux d'entraînement (cache .npz) et échantillonnage de fenêtres.

  synthétique : features + vérité terrain complète (atome de pose, groupe, keypoints 3D, taille)
  réel        : features + groupe d'activité faible par bin (labels.py), keypoints inconnus

    python -m skeleton.dataset --n-per-class 500     # (re)crée skeleton/cache/*.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config as cfg
from . import labels as L
from . import synth
from .features import raw_to_complex, sim_to_rx, take_features

CACHE = os.path.join(os.path.dirname(__file__), "cache")
T_TAKE = int(cfg.TAKE_S * cfg.FS)


def _sim_job(args):
    seed, activity, env_seed = args[:3]
    ideal = len(args) > 3 and args[3]
    rng = np.random.default_rng(seed)
    env = synth.Environment(env_seed)
    rx_out, lab, info = synth.simulate_take(activity, rng, env, ideal=ideal)
    t0 = lab["t0_ms"] + 5
    X, _, _ = take_features(sim_to_rx(rx_out), t_start_ms=t0, duration=cfg.TAKE_S)
    # labels au centre de chaque bin
    i = np.minimum(np.round((np.arange(T_TAKE) + 0.5) / cfg.FS * cfg.TX_RATE_HZ).astype(int), len(lab["t"]) - 1)
    return dict(X=X.astype(np.float16), atom=lab["atom"][i], group=lab["group"][i],
                kp=lab["kp_body"][i].astype(np.float16), height=np.float32(info.get("height_m", 0.0)),
                act=synth.ACTIVITIES.index(activity), env=env_seed)


def build_synth(n_per_class, seed=0, n_env=40, workers=None, path=None, ideal=False):
    rng = np.random.default_rng(seed)
    jobs = []
    for act in synth.ACTIVITIES:
        n = n_per_class // 3 if act == "vide" else n_per_class
        for _ in range(n):
            jobs.append((int(rng.integers(2 ** 31)), act, 10_000 + int(rng.integers(n_env)), ideal))
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        res = list(ex.map(_sim_job, jobs, chunksize=8))
    out = {k: np.stack([r[k] for r in res]) for k in res[0]}
    path = path or os.path.join(CACHE, "synth.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def _aug_job(args):
    """k copies augmentées d'une vraie prise (augment_real.py), features + étiquettes."""
    from .augment_real import augment_take
    from .features import load_take
    take, activity, t_fall, partners, k, seed = args
    rng = np.random.default_rng(seed)
    rx = load_take(os.path.join(L.REAL_ROOT, take))
    out = []
    for _ in range(k):
        partner = load_take(os.path.join(L.REAL_ROOT, partners[rng.integers(len(partners))])) if partners else None
        rx_a, tf, _ = augment_take(rx, activity, rng, None if np.isnan(t_fall) else t_fall, partner)
        if len(rx_a) == 0:
            continue
        X, _, _ = take_features(rx_a, duration=cfg.TAKE_S)
        out.append((X.astype(np.float16), L.frame_groups(activity, len(X), tf)))
    return out


def build_real_cache(path=None, k_aug=4, seed=0):
    """Vraies prises (is_aug=False) + k_aug copies augmentées par prise (is_aug=True, orig = indice source)."""
    data = L.build_real()
    X = [d["X"].astype(np.float16) for d in data]
    y = [d["y"] for d in data]
    orig = list(range(len(data)))
    is_aug = [False] * len(data)
    if k_aug:
        rng = np.random.default_rng(seed)
        jobs = []
        for d in data:
            partners = [e["take"] for e in data if e["activity"] == d["activity"] and e["take"] != d["take"]]
            jobs.append((d["take"], d["activity"], np.nan if d["t_fall"] is None else d["t_fall"], partners,
                         k_aug, int(rng.integers(2 ** 31))))
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            for i, res in enumerate(ex.map(_aug_job, jobs)):
                for Xa, ya in res:
                    X.append(Xa)
                    y.append(ya)
                    orig.append(i)
                    is_aug.append(True)
    path = path or os.path.join(CACHE, "real_aug.npz" if k_aug else "real.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o = np.array(orig)
    np.savez_compressed(path, X=np.stack(X), y=np.stack(y), orig=o, is_aug=np.array(is_aug),
                        take=np.array([data[i]["take"] for i in o]),
                        activity=np.array([data[i]["activity"] for i in o]),
                        t_fall=np.array([np.nan if data[i]["t_fall"] is None else data[i]["t_fall"] for i in o]))
    return path


def load(name):
    return dict(np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=False))


# --------------------------------------------------------------------------
# Fenêtres + augmentation
# --------------------------------------------------------------------------
def augment(X, rng, p_rx_drop=0.12):
    """X (B, T, N_IN) float32 : coupure de RX, gain sur 'dyn', bruit."""
    B = X.shape[0]
    fp, nb = cfg.FEAT_PER_RX, cfg.N_BANDS
    for k in range(len(cfg.RX_NAMES)):
        sl = slice(k * fp, (k + 1) * fp)
        drop = rng.random(B) < p_rx_drop
        X[drop, :, sl] = 0
        g = rng.uniform(0.75, 1.35, (B, 1, 1))
        X[:, :, k * fp:k * fp + nb] *= g
    X[..., :] += rng.normal(0, 0.05, X.shape).astype(np.float32) * (X != 0)
    return X


def windows(n_take, T_win, batch, rng):
    """Indices (prise, début) aléatoires."""
    return rng.integers(0, n_take, batch), rng.integers(0, T_TAKE - T_win + 1, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-aug", type=int, default=4, help="copies augmentées par vraie prise")
    ap.add_argument("--real-only", action="store_true", help="ne reconstruit que le cache réel")
    ap.add_argument("--seq", action="store_true", help="cache de séquences multi-états (seq_holdout.npz + seq.npz)")
    ap.add_argument("--ideal", action="store_true",
                    help="cache synthétique 'acquisition parfaite' (synth_ideal.npz, étude de faisabilité)")
    a = ap.parse_args()
    if a.seq:
        print("séquences (sans prises mises de côté) ->", build_seq_cache(exclude_holdout=True))
        print("séquences (toutes prises)              ->", build_seq_cache(exclude_holdout=False, seed=1))
        return
    if a.ideal:
        print("synthétique idéal ->", build_synth(a.n_per_class, a.seed, ideal=True,
                                                   path=os.path.join(CACHE, "synth_ideal.npz")))
        return
    if not a.real_only:
        print("synthétique ->", build_synth(a.n_per_class, a.seed))
    print("réel        ->", build_real_cache(k_aug=a.k_aug, seed=a.seed))



# --------------------------------------------------------------------------
# Séquences multi-états (apprendre que s'asseoir / s'arrêter n'est PAS une chute)
# --------------------------------------------------------------------------
SEQ_S = 16.0
SEQ_T = int(SEQ_S * cfg.FS)


def _random_script(rng):
    states = ["debout", "assis", "marche"]
    script, prev, tot = [], None, 0.0
    while tot < SEQ_S + 1:
        if tot > 9 and rng.random() < 0.5:
            script.append(("chute", float(rng.uniform(4, 6))))
            tot += script[-1][1]
            break
        s = rng.choice([x for x in states if x != prev])
        d = float(rng.uniform(2.5, 5.0))
        script.append((s, d))
        prev, tot = s, tot + d
    return script


def _seq_labels(truth, ignore=0.3):
    t = (np.arange(SEQ_T) + 0.5) / cfg.FS
    y = np.full(SEQ_T, -1, np.int64)
    for a, b, g in truth:
        y[(t >= a) & (t < b)] = cfg.GROUPS.index(g)
    for a, _, _ in truth[1:]:
        y[np.abs(t - a) < ignore] = -1
    return y


def _seq_job(args):
    from . import sequences as S
    seed, source, takes = args
    rng = np.random.default_rng(seed)
    script = _random_script(rng)
    if source == "reel":
        seq = S.real_montage(script, takes, seed=seed, blend_s=float(rng.uniform(0.8, 1.5)))
    else:
        seq = S.simulate_sequence(script, seed=seed, env_seed=int(10_000 + rng.integers(40)))
    rx = {n: (ts, raw_to_complex(r)) for n, (ts, r) in seq["rx"].items() if len(ts) >= 5}
    X, _, _ = take_features(rx, t_start_ms=seq["t0_ms"], duration=SEQ_S)
    return X.astype(np.float16), _seq_labels(seq["truth"]), source == "reel"


def build_seq_cache(n_real=600, n_sim=300, seed=0, exclude_holdout=True, path=None):
    from .holdout import holdout_takes
    from .labels import real_takes
    idx = real_takes()
    held = {t for v in holdout_takes().values() for t in v} if exclude_holdout else set()
    takes = {a: sorted(t for t in g.chemin if t not in held) for a, g in idx.groupby("activite")}
    rng = np.random.default_rng(seed)
    jobs = [(int(rng.integers(2 ** 31)), "reel", takes) for _ in range(n_real)] + \
           [(int(rng.integers(2 ** 31)), "simule", None) for _ in range(n_sim)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(_seq_job, jobs, chunksize=4))
    path = path or os.path.join(CACHE, "seq_holdout.npz" if exclude_holdout else "seq.npz")
    np.savez_compressed(path, X=np.stack([r[0] for r in res]), y=np.stack([r[1] for r in res]),
                        is_real=np.array([r[2] for r in res]))
    return path


if __name__ == "__main__":
    main()
