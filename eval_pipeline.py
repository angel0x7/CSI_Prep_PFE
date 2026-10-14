#!/usr/bin/env python3
"""
Évalue l'apport du prétraitement csi_preprocess.py sur le dataset Activity (FA/GU/LD/SD/WA).

Trois variantes du même signal sont comparées avec exactement les mêmes features et le même
classifieur, validation croisée "leave-subject-out" (aucun sujet commun train/test) :
  A. brut        : out_raw/  amp         (sélection de sous-porteuses + rééchantillonnage seulement)
  B. Hampel      : out_full/ amp_clean   (A + suppression des pics)
  C. complet     : out_full/ amp         (B + passe-bande Butterworth 0.3-10 Hz)
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

FS = 50
WIN, STEP, TRIM = 150, 75, 50          # fenêtres 3 s, pas 1.5 s, on enlève 1 s aux bords (effets de bord du filtre)
BANDS = [(0.3, 1), (1, 2), (2, 4), (4, 7), (7, 10), (10, 25)]
CLASSES = ["FA", "GU", "LD", "SD", "WA"]


def window_feats(w):
    n = len(w)
    w = w - w.mean(0)
    std = w.std(0)
    f = [std.mean(), np.median(std), np.abs(np.diff(w, axis=0)).mean(),
         (w.max(0) - w.min(0)).mean(), kurtosis(w, axis=0).mean(), w.mean(1).std()]
    e = np.linalg.svd(w, compute_uv=False) ** 2
    f += [e[0] / e.sum(), e[:3].sum() / e.sum()]
    P = (np.abs(np.fft.rfft(w * np.hanning(n)[:, None], axis=0)) ** 2).mean(1)
    fr = np.fft.rfftfreq(n, 1 / FS)
    tot = P[(fr >= 0.3) & (fr <= 10)].sum() + 1e-12
    for lo, hi in BANDS:
        s = P[(fr >= lo) & (fr < hi)].sum()
        f += [np.log10(s + 1e-12), s / tot]
    return f


def extract(root: Path, key: str):
    X, y, g, fid = [], [], [], []
    for i, p in enumerate(sorted(root.rglob("*.npz"))):
        act, subj = p.parts[-3], p.parts[-2]
        a = np.load(p)[key][TRIM:-TRIM]
        for s in range(0, len(a) - WIN + 1, STEP):
            X.append(window_feats(a[s:s + WIN].astype(np.float64)))
            y.append(CLASSES.index(act)); g.append(subj); fid.append(i)
    return np.array(X), np.array(y), np.array(g), np.array(fid)


def evaluate(X, y, g, make_model, n_splits=7):
    P = np.zeros((len(y), len(CLASSES)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        m = make_model().fit(X[tr], y[tr])
        P[te] = m.predict_proba(X[te])
    pred = P.argmax(1)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro"),
                auc_fall=roc_auc_score(y == 0, P[:, 0]), cm=confusion_matrix(y, pred), P=P)


if __name__ == "__main__":
    W = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    variants = {"A brut": (W / "out_raw", "amp"),
                "B Hampel": (W / "out_full", "amp_clean"),
                "C complet": (W / "out_full", "amp")}
    models = {"RandomForest": lambda: RandomForestClassifier(150, random_state=0, n_jobs=-1),
              "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    res = {}
    for name, (root, key) in variants.items():
        X, y, g, _ = extract(root, key)
        print(f"{name}: {X.shape[0]} fenêtres, {len(set(g))} sujets, {X.shape[1]} features", flush=True)
        for mn, mk in models.items():
            r = evaluate(X, y, g, mk)
            res[(name, mn)] = r
            print(f"   {mn:13s} acc={r['acc']:.3f}  F1macro={r['f1']:.3f}  AUC(chute vs reste)={r['auc_fall']:.3f}", flush=True)
    print("\nMatrice de confusion, variante C + RandomForest (lignes=vrai, colonnes=prédit)", CLASSES)
    print(res[("C complet", "RandomForest")]["cm"])
    print("chance niveau classe équilibrée ≈ %.3f" % (1 / len(CLASSES)))
