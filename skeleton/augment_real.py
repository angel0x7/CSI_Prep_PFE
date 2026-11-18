# -*- coding: utf-8 -*-
"""
Augmentation des VRAIES prises : génère de nouvelles prises crédibles à partir des CSV réels.

Chaque transformation correspond à un phénomène physique plausible (pas de bruit arbitraire) :
  - étirement temporel x0.9–1.15            personne plus lente / plus rapide
  - inversion du temps                       (debout, assis, marche seulement : une chute ne se rembobine pas)
  - décalage du début (chute)                chute plus tôt / plus tard dans la prise
  - pertes supplémentaires en rafales        Wi-Fi plus dégradé (modèle de Gilbert-Elliott)
  - gain par RX + bruit + re-quantification  autre réglage AGC / autre niveau de bruit
  - ondulation spectrale douce par RX        antenne / placement légèrement différents
  - décalage d'horloge par RX (±15 ms)       synchronisation imparfaite
  - recombinaison (debout, assis, marche)    forme statique du canal de la prise A
                                              x variations temporelles de la prise B (même classe, même RX)
                                              = même activité, autre position / autre séance

Les copies restent rattachées à leur prise d'origine (champ 'source') : en validation croisée,
une prise et ses copies sont toujours du même côté (sinon fuite et scores faussés).

    python -m skeleton.augment_real --k 4                 # -> acquisition/data/DATASET_AUG
    (copie du DATASET réel + k prises augmentées par prise p1 utilisable)
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from . import config as cfg
from .features import load_take

REAL_ROOT = os.path.join(os.path.dirname(__file__), "..", "acquisition", "data", "DATASET")
NO_REVERSE = {"chute"}
NO_RECOMBINE = {"chute"}


def _gilbert_keep(rng, n, p, burst=3.0):
    if p <= 0:
        return np.ones(n, bool)
    q_bg = 1.0 / burst
    q_gb = p * q_bg / max(1 - p, 1e-3)
    keep = np.empty(n, bool)
    bad = False
    r = rng.random(n)
    for i in range(n):
        bad = (r[i] >= q_bg) if bad else (r[i] < q_gb)
        keep[i] = not bad
    return keep


def _smooth_ripple(rng, amp):
    x = np.linspace(-1, 1, cfg.N_SC)
    c = rng.normal(0, 1, 4)
    ph = rng.uniform(0, 2 * np.pi)
    return 1 + amp * sum(c[i] * np.cos((i + 1) * np.pi * x / 2 + ph) for i in range(3)) / 2


def augment_take(rx, activity, rng, t_fall=None, partner=None, duration=cfg.TAKE_S):
    """rx = {RXn: (t_ms int64, H complexe (n,52))} -> (rx_aug, t_fall_aug, meta)."""
    t0 = min(int(v[0][0]) for v in rx.values())
    t1 = max(int(v[0][-1]) for v in rx.values())
    meta = {}
    s = float(rng.uniform(0.9, 1.15))
    rev = activity not in NO_REVERSE and rng.random() < 0.5
    shift = float(rng.uniform(-0.6, 0.6)) if activity == "chute" else 0.0
    if t_fall is not None:                                   # garder la chute dans [0.5 s, 5 s]
        shift = float(np.clip(shift, t_fall * s - 5.0, t_fall * s - 0.5))
    recomb = partner is not None and activity not in NO_RECOMBINE and rng.random() < 0.4
    meta.update(stretch=round(s, 3), reversed=bool(rev), shift_s=round(shift, 3), recombined=bool(recomb))

    out = {}
    for name, (t, H) in rx.items():
        tt = (t - t0).astype(float)
        if rev:
            tt = (t1 - t0) - tt[::-1]
            H = H[::-1]
        tt = tt * s - shift * 1000.0
        if recomb and name in partner:
            # forme statique de la prise courante, dynamique relative de la prise partenaire
            tp, Hp = partner[name]
            Ap = np.abs(Hp)
            rel = Ap / np.maximum(Ap.mean(0, keepdims=True), 1e-3)          # variations temporelles de B
            tpp = (tp - tp[0]).astype(float) * s
            idx = np.clip(np.searchsorted(tpp, tt), 0, len(tpp) - 1)
            base = np.abs(H).mean(0, keepdims=True)                         # forme moyenne de A
            H = base * rel[idx] * np.exp(1j * np.angle(Hp[idx]))
        # matériel : gain, ondulation spectrale, bruit, re-quantification
        g = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
        H = H * g * _smooth_ripple(rng, rng.uniform(0.0, 0.12))[None, :]
        H = H + rng.uniform(0.2, 0.6) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        H = np.round(H.real) + 1j * np.round(H.imag)
        H = np.clip(H.real, -127, 127) + 1j * np.clip(H.imag, -127, 127)
        tt = tt + rng.normal(0, 8) + rng.normal(0, 0.8, len(tt))           # horloge du RX
        keep = (tt >= 0) & (tt < duration * 1000)
        keep &= _gilbert_keep(rng, len(tt), float(rng.uniform(0.0, 0.25)), rng.uniform(2, 5))
        o = np.argsort(tt[keep])
        tk = np.round(tt[keep][o]).astype(np.int64) + t0
        if len(tk) >= 5:
            out[name] = (tk, H[keep][o])
    tf = None if t_fall is None else t_fall * s - shift
    return out, tf, meta


def rx_to_csv_rows(t, H):
    full = np.zeros((len(t), 64), complex)
    full[:, cfg.SC_COLS] = H
    raw = np.empty((len(t), 128), np.int64)
    raw[:, 0::2] = np.round(full.imag)
    raw[:, 1::2] = np.round(full.real)
    return np.concatenate([t[:, None], raw], 1)


def write_dataset(out_root, k=4, persons=(1,), seed=0, copy_real=True):
    from .labels import fall_time
    from .features import take_features
    rng = np.random.default_rng(seed)
    idx = pd.read_csv(os.path.join(REAL_ROOT, "index.csv"))
    if copy_real:
        shutil.copytree(REAL_ROOT, out_root, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.txt", "skeleton_pred*"))
    rows = [dict(chemin=r.chemin, activite=r.activite, personnes=r.personnes, prise=r.prise,
                 source=r.chemin, augmentee=False, qualite=r.qualite, augmentation="")
            for _, r in idx.iterrows()]
    use = idx[idx.personnes.isin(persons) & (idx.qualite != "inutilisable")]
    takes = {r.chemin: load_take(os.path.join(REAL_ROOT, r.chemin)) for _, r in use.iterrows()}
    for _, r in use.iterrows():
        rx = takes[r.chemin]
        tf = None
        if r.activite == "chute":
            tf = fall_time(take_features(rx, duration=cfg.TAKE_S)[0])[0]
        same = [c for c in use[(use.activite == r.activite) & (use.personnes == r.personnes)].chemin if c != r.chemin]
        for j in range(k):
            partner = takes[same[rng.integers(len(same))]] if same else None
            rx_a, tf_a, meta = augment_take(rx, r.activite, rng, tf, partner)
            rel = "%s_aug%d" % (r.chemin, j + 1)
            d = os.path.join(out_root, rel)
            os.makedirs(d, exist_ok=True)
            for name, (t, H) in rx_a.items():
                np.savetxt(os.path.join(d, name + ".csv"), rx_to_csv_rows(t, H), fmt="%d", delimiter=",")
            meta.update(source=r.chemin, t_fall_estime_s=None if tf_a is None else round(tf_a, 3))
            with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"source": "skeleton.augment_real (AUGMENTATION d'une vraie prise)",
                           "dir": rel, "augmentation": meta,
                           "receivers": {n: {"frames": int(len(t))} for n, (t, _) in rx_a.items()}},
                          f, indent=2, ensure_ascii=False)
            rows.append(dict(chemin=rel, activite=r.activite, personnes=r.personnes,
                             prise=os.path.basename(rel), source=r.chemin, augmentee=True,
                             qualite="augmentee", augmentation=json.dumps(meta)))
        print(r.chemin, "x%d" % k, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "index.csv"), index=False)
    with open(os.path.join(out_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(README % dict(k=k))


README = """# DATASET_AUG : vraies prises + prises augmentées

Généré par `python -m skeleton.augment_real --k %(k)d` (reproductible, seed fixée ; non versionné sur git).

- Les dossiers `prise_NNN` sont des **copies exactes** du `DATASET` réel (lire son README : pertes Wi-Fi).
- Les dossiers `prise_NNN_augK` sont des **prises augmentées** générées à partir de la prise réelle `prise_NNN`
  (étirement temporel, inversion, décalage de la chute, pertes supplémentaires, gain/bruit/quantification,
  ondulation spectrale, désynchronisation des RX, recombinaison statique/dynamique entre prises de même classe).
  Le détail de chaque prise est dans son `summary.json` (champ `augmentation`).
- `index.csv` : colonne `source` = prise réelle d'origine, `augmentee` = True/False.

**Règle d'évaluation** : ne jamais mettre une prise augmentée dans le test, ni une prise réelle en test alors
que ses copies `_augK` sont en entraînement. Découper par `source`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("acquisition", "data", "DATASET_AUG"))
    ap.add_argument("--k", type=int, default=4, help="prises augmentées par prise réelle")
    ap.add_argument("--persons", type=int, nargs="*", default=[1])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_dataset(a.out, a.k, tuple(a.persons), a.seed)


if __name__ == "__main__":
    main()
