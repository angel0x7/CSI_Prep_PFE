# -*- coding: utf-8 -*-
"""
Entraînement + évaluation du modèle squelette.

    python -m skeleton.train cv       # validation croisée 5 plis sur les vraies prises + ablations
    python -m skeleton.train final    # modèle final (synthétique + toutes les vraies prises)
                                       -> skeleton/checkpoints/skeleton_tcn.pt

Évaluation honnête :
  - sur le RÉEL on ne peut mesurer que l'activité (par bin et par prise) et l'instant de chute
    (référence = instant estimé par labels.fall_time) : pas de vérité terrain squelette ;
  - l'erreur de squelette (MPJPE, PCK) est mesurée sur des prises SIMULÉES dans des
    'séances' (mobilier / matériel) jamais vues à l'entraînement.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .dataset import T_TAKE, augment, load
from .model import SkeletonTCN, n_params

HERE = os.path.dirname(__file__)
G = {g: i for i, g in enumerate(cfg.GROUPS)}
REAL_GROUPS = ["debout", "assis", "marche", "chute", "au_sol"]
WARMUP = 20            # bins ignorés en début de fenêtre (contexte causal incomplet)
T_WIN = int(cfg.WIN_S * cfg.FS)


def static_mask(use_static):
    """use_static : True / 'all' = toutes les features ; False / 'no_static' = sans forme statique ;
    'dyn_only' = uniquement les variations rapides (ni forme statique, ni niveau, ni masque de pertes :
    ces trois-là peuvent porter l'empreinte de la séance d'enregistrement plutôt que la posture)."""
    mode = {True: "all", False: "no_static"}.get(use_static, use_static)
    m = np.ones(cfg.N_IN, np.float32)
    fp, nb = cfg.FEAT_PER_RX, cfg.N_BANDS
    for k in range(len(cfg.RX_NAMES)):
        if mode in ("no_static", "dyn_only"):
            m[k * fp + nb:k * fp + 2 * nb] = 0
        if mode == "dyn_only":
            m[k * fp + 2 * nb:(k + 1) * fp] = 0
    return m


# --------------------------------------------------------------------------
CLIP_DILATIONS = (1, 2, 4, 8, 16, 32)


def train_model(syn, real, idx_syn, idx_real, use_synth=True, use_real=True, use_static=True,
                steps=1500, seed=0, verbose=False, clip=False, seq=None, init=None, lr=2e-3):
    """clip=False : modèle temps réel (causal, fenêtres de 4 s, 1 s de démarrage ignorée) ;
    clip=True  : modèle 'prise entière' (non causal, prises de 8 s complètes, aucune zone ignorée).
    seq = cache de séquences multi-états (dataset.build_seq_cache) : transitions normales
          (s'asseoir, s'arrêter...) étiquetées, pour que le modèle ne les prenne pas pour des chutes.
    init = poids de départ (state_dict) pour AFFINER un modèle existant (normalisation conservée)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cm = static_mask(use_static)
    model = SkeletonTCN(causal=not clip, dilations=CLIP_DILATIONS) if clip else SkeletonTCN()
    T_WIN = T_TAKE if clip else globals()["T_WIN"]
    WARMUP = 0 if clip else globals()["WARMUP"]
    model.warmup = WARMUP
    parts = []
    if use_synth:
        parts.append(syn["X"][idx_syn[:400]].reshape(-1, cfg.N_IN))
    if use_real:
        parts.append(real["X"][np.sort(rng.permutation(idx_real)[:600])].reshape(-1, cfg.N_IN))
    allx = np.concatenate(parts).astype(np.float32) * cm
    if init is not None:
        model.load_state_dict(init)
    else:
        model.set_norm(allx.mean(0), allx.std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    # poids de classes pour le réel (marche sous-représentée)
    yr = real["y"][idx_real]
    cnt = np.array([(yr == G[g]).sum() for g in cfg.GROUPS], float)
    wcls = torch.tensor(np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / (cnt > 0).sum(), 0.0), dtype=torch.float32)

    Bs, Br = (48 if use_synth else 0), (16 if use_real else 0)
    if not use_synth:
        Br = 48
    if clip:                                     # fenêtres 2x plus longues -> lots plus petits
        Bs, Br = int(Bs * 0.6), max(8, int(Br * 0.75))
    model.train()
    t0 = time.time()
    for step in range(steps):
        loss = 0.0
        if Bs:
            ti = idx_syn[rng.integers(0, len(idx_syn), Bs)]
            st = rng.integers(0, T_TAKE - T_WIN + 1, Bs)
            sl = np.stack([np.arange(s, s + T_WIN) for s in st])
            X = syn["X"][ti[:, None], sl].astype(np.float32) * cm
            X = torch.from_numpy(augment(X, rng))
            atom = torch.from_numpy(syn["atom"][ti[:, None], sl][:, WARMUP:].astype(np.int64))
            grp = torch.from_numpy(syn["group"][ti[:, None], sl][:, WARMUP:].astype(np.int64))
            kp = torch.from_numpy(syn["kp"][ti[:, None], sl][:, WARMUP:].astype(np.float32))
            hgt = torch.from_numpy(np.repeat(syn["height"][ti][:, None], T_WIN - WARMUP, 1))
            out = model(X)
            lg = out["atom_logits"][:, WARMUP:]
            l_atom = F.cross_entropy(lg.reshape(-1, cfg.N_ATOMS), atom.reshape(-1), label_smoothing=0.05)
            l_grp = F.nll_loss(out["group_logp"][:, WARMUP:].reshape(-1, len(cfg.GROUPS)), grp.reshape(-1))
            pres = (atom != 0).float()
            err = (out["kp"][:, WARMUP:] - kp).norm(dim=-1).mean(-1)
            l_kp = (err * pres).sum() / pres.sum().clamp(min=1)
            l_h = ((out["height"][:, WARMUP:] - hgt).abs() * pres).sum() / pres.sum().clamp(min=1)
            loss = loss + l_atom + l_grp + 3 * l_kp + 2 * l_h
        if seq is not None:
            Bq = 16
            ti = rng.integers(0, len(seq["X"]), Bq)
            st = rng.integers(0, seq["X"].shape[1] - T_WIN + 1, Bq)
            sl = np.stack([np.arange(s0, s0 + T_WIN) for s0 in st])
            Xq = torch.from_numpy(augment(seq["X"][ti[:, None], sl].astype(np.float32) * cm, rng))
            yq = torch.from_numpy(seq["y"][ti[:, None], sl][:, WARMUP:])
            oq = model(Xq)
            loss = loss + F.nll_loss(oq["group_logp"][:, WARMUP:].reshape(-1, len(cfg.GROUPS)), yq.reshape(-1),
                                     weight=wcls, ignore_index=-1)
        if Br:
            ti = idx_real[rng.integers(0, len(idx_real), Br)]
            st = rng.integers(0, T_TAKE - T_WIN + 1, Br)
            sl = np.stack([np.arange(s, s + T_WIN) for s in st])
            X = torch.from_numpy(augment(real["X"][ti[:, None], sl].astype(np.float32) * cm, rng))
            y = torch.from_numpy(real["y"][ti[:, None], sl][:, WARMUP:])
            out = model(X)
            l_real = F.nll_loss(out["group_logp"][:, WARMUP:].reshape(-1, len(cfg.GROUPS)), y.reshape(-1),
                                weight=wcls, ignore_index=-1)
            loss = loss + l_real
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step % 250 == 0 or step == steps - 1):
            print("  step %4d  loss %.3f  (%.0fs)" % (step, loss.item(), time.time() - t0), flush=True)
    model.eval()
    model.channel_mask = torch.from_numpy(cm)
    return model


@torch.no_grad()
def predict(model, X):
    """X (N, T, N_IN) -> sorties numpy sur la prise entière (causal)."""
    cm = getattr(model, "channel_mask", torch.ones(cfg.N_IN))
    out = model(torch.from_numpy(X.astype(np.float32)) * cm)
    return {k: v.numpy() for k, v in out.items()}


# --------------------------------------------------------------------------
def take_decision(gp, warmup=WARMUP):
    """gp (T,6) log-probas de groupes -> activité de la prise (même règle que le dashboard)."""
    a = gp[warmup:].argmax(-1)
    fall = np.isin(a, [G["chute"], G["au_sol"]]).mean()
    if fall >= 0.2:
        return "chute"
    c = np.bincount(a, minlength=len(cfg.GROUPS))
    c[[G["chute"], G["au_sol"]]] = 0
    return cfg.GROUPS[int(c.argmax())]


def fall_onset(gp, thr=0.5, warmup=WARMUP):
    p = np.exp(gp[:, G["chute"]]) + np.exp(gp[:, G["au_sol"]])
    p = np.convolve(p, np.ones(5) / 5, mode="same")
    i = np.nonzero(p[warmup // 2:] > thr)[0]
    return None if len(i) == 0 else (i[0] + warmup // 2) / cfg.FS


def eval_real(model, real, idx):
    WARMUP = getattr(model, "warmup", globals()["WARMUP"])
    out = predict(model, real["X"][idx])
    gp = out["group_logp"]
    y = real["y"][idx][:, WARMUP:]
    pred = gp[:, WARMUP:].argmax(-1)
    m = y >= 0
    labs = [G[g] for g in REAL_GROUPS] + [G["vide"]]
    conf = np.zeros((len(labs), len(labs)), int)
    for a, b in zip(y[m], pred[m]):
        conf[labs.index(a), labs.index(b)] += 1
    take_true = list(real["activity"][idx])
    take_pred = [take_decision(g, WARMUP) for g in gp]
    fe = []
    for g, tf, act in zip(gp, real["t_fall"][idx], take_true):
        if act == "chute":
            on = fall_onset(g, warmup=WARMUP)
            fe.append(np.nan if on is None else on - (tf - 0.4))
    bacc = np.mean([conf[i, i] / max(conf[i].sum(), 1) for i in range(len(REAL_GROUPS)) if conf[i].sum()])
    return dict(bin_acc=float((pred[m] == y[m]).mean()), bin_bal_acc=float(bacc), confusion=conf.tolist(),
                take_true=take_true, take_pred=take_pred,
                take_acc=float(np.mean([a == b for a, b in zip(take_true, take_pred)])),
                fall_onset_err_s=[float(x) for x in fe])


def eval_synth(model, syn, idx):
    WARMUP = getattr(model, "warmup", globals()["WARMUP"])
    out = predict(model, syn["X"][idx])
    kp = syn["kp"][idx].astype(np.float32)[:, WARMUP:]
    pk = out["kp"][:, WARMUP:]
    atom = syn["atom"][idx][:, WARMUP:]
    grp = syn["group"][idx][:, WARMUP:]
    pres = atom != 0
    err = np.linalg.norm(pk - kp, axis=-1)                           # (N,T,17)
    torso = np.linalg.norm(kp[..., 5, :] - kp[..., 12, :], axis=-1)  # épaule G - hanche D
    pck = (err < 0.2 * torso[..., None])[pres].mean()
    res = dict(mpjpe_mm=float(err[pres].mean() * 1000), pck_0_2=float(pck),
               group_acc=float((out["group_logp"][:, WARMUP:].argmax(-1) == grp).mean()))
    for g in cfg.GROUPS[1:]:
        mm = pres & (grp == G[g])
        if mm.any():
            res["mpjpe_mm_" + g] = float(err[mm].mean() * 1000)
    # référence naïve : toujours le squelette 'debout' moyen
    base = np.linalg.norm(SkeletonTCN().atoms[1].numpy()[None, None] - kp, axis=-1)
    res["mpjpe_mm_baseline_debout"] = float(base[pres].mean() * 1000)
    return res


def synth_split(syn, n_test_env=6, seed=0):
    envs = np.unique(syn["env"])
    rng = np.random.default_rng(seed)
    test_env = rng.choice(envs, n_test_env, replace=False)
    te = np.isin(syn["env"], test_env)
    return np.nonzero(~te)[0], np.nonzero(te)[0]


def summarize(rs):
    def ms(k):
        v = [r[k] for r in rs]
        return "%.3f ± %.3f" % (np.mean(v), np.std(v))
    fe = np.concatenate([r["fall_onset_err_s"] for r in rs])
    return {"bin_acc": ms("bin_acc"), "bin_bal_acc": ms("bin_bal_acc"), "take_acc": ms("take_acc"),
            "fall_detected": "%d/%d" % (np.isfinite(fe).sum(), len(fe)),
            "fall_onset_abs_err_median_s": float(np.nanmedian(np.abs(fe))) if np.isfinite(fe).any() else None,
            "confusion_sum": np.sum([r["confusion"] for r in rs], 0).tolist()}


CONFIGS = {
    "reel_seul": dict(use_synth=False, use_real=True, use_static="all"),
    "synth_seul_zero_shot": dict(use_synth=True, use_real=False, use_static="all"),
    "synth+reel": dict(use_synth=True, use_real=True, use_static="all"),
    "synth+reel_sans_statique": dict(use_synth=True, use_real=True, use_static="no_static"),
    "synth+reel_dyn_seul": dict(use_synth=True, use_real=True, use_static="dyn_only"),
    "reel_seul_dyn_seul": dict(use_synth=False, use_real=True, use_static="dyn_only"),
    "synth+reel_prise_entiere": dict(use_synth=True, use_real=True, use_static="all", clip=True),
}


def real_split(real, tr, te_orig):
    """Indices d'entraînement = prises d'origine du pli + leurs copies augmentées ;
    test = prises d'origine seulement (jamais de copie augmentée en test)."""
    orig = real.get("orig", np.arange(len(real["X"])))
    return np.nonzero(np.isin(orig, tr))[0], te_orig


def run_cv(steps, folds=5, only=None, real_cache="real"):
    syn, real = load("synth"), load(real_cache)
    tr_s, te_s = synth_split(syn)
    configs = {k: v for k, v in CONFIGS.items() if only is None or k in only}
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    is_aug = real.get("is_aug", np.zeros(len(real["X"]), bool))
    originals = np.nonzero(~is_aug)[0]
    results = {}
    for name, c in configs.items():
        rs, ss = [], []
        print("==", name, "(cache réel : %s, %d prises dont %d augmentées)" % (real_cache, len(is_aug), is_aug.sum()),
              flush=True)
        for k, (tr, te) in enumerate(skf.split(originals, real["activity"][originals])):
            idx_tr, idx_te = real_split(real, originals[tr], originals[te])
            m = train_model(syn, real, tr_s, idx_tr, steps=steps, seed=k, **c)
            r = eval_real(m, real, idx_te)
            rs.append(r)
            if c["use_synth"]:
                ss.append(eval_synth(m, syn, te_s))
            print("  pli %d : bin_acc %.3f  take_acc %.3f" % (k, r["bin_acc"], r["take_acc"]), flush=True)
        results[name] = {"real": summarize(rs)}
        if ss:
            results[name]["synth_test"] = {k: float(np.mean([s[k] for s in ss])) for k in ss[0]}
        print(json.dumps(results[name], indent=1), flush=True)
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        tag = name.replace("+", "_") + ("" if real_cache == "real" else "__" + real_cache)
        with open(os.path.join(HERE, "results", "cv_%s.json" % tag), "w", encoding="utf-8") as f:
            json.dump(dict(config=name, real_cache=real_cache, steps=steps, folds=folds, groups=REAL_GROUPS + ["vide"],
                           **results[name]), f, indent=2, ensure_ascii=False)


def run_final(steps, features="all", real_cache="real_aug", clip=False):
    syn, real = load("synth"), load(real_cache)
    tr_s, te_s = synth_split(syn)
    seq = load("seq") if (not clip and os.path.exists(os.path.join(HERE, "cache", "seq.npz"))) else None
    print("séquences multi-états :", "oui (%d)" % len(seq["X"]) if seq is not None else "non", flush=True)
    m = train_model(syn, real, tr_s, np.arange(len(real["X"])), steps=steps, seed=0, verbose=True,
                    use_static=features, clip=clip, seq=seq)
    s = eval_synth(m, syn, te_s)
    print(json.dumps(s, indent=1))
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    path = os.path.join(HERE, "checkpoints", "skeleton_clip.pt" if clip else "skeleton_tcn.pt")
    torch.save({"state_dict": m.state_dict(), "channel_mask": m.channel_mask, "config": {
        "fs": cfg.FS, "groups": cfg.GROUPS, "atom_group": cfg.ATOM_GROUP, "rx": cfg.RX_NAMES,
        "n_in": cfg.N_IN, "steps": steps, "features": features,
        "causal": m.causal, "dilations": list(m.dilations), "warmup_bins": m.warmup}, "synth_test": s}, path)
    print("->", path, "(%d paramètres)" % n_params(m))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "final"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="configurations à lancer (voir CONFIGS)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--features", default="all", help="final : all | no_static | dyn_only")
    ap.add_argument("--real-cache", default=None, help="real (sans augmentation) | real_aug")
    ap.add_argument("--clip", action="store_true", help="final : modèle 'prise entière' (non causal, 8 s)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.mode == "cv":
        run_cv(a.steps, a.folds, a.only, a.real_cache or "real")
    else:
        run_final(a.steps, a.features, a.real_cache or "real_aug", a.clip)
