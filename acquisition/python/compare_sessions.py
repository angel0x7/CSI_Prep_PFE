# -*- coding: utf-8 -*-
"""
WISEE - Comparaison de plusieurs sessions CSI (ex : piece vide / marche / assis immobile).

Meme traitement que analyze_session.py pour chaque session (partie HT-LTF, horloge de l'ESP32,
reechantillonnage a fs Hz, indice de mouvement = ecart-type glissant 1 s). Les sous-porteuses
valides sont choisies sur la PREMIERE session (la reference : piece vide) puis appliquees a toutes.

Produit dans reports/<nom>/ : motion_compare.png, motion_boxplot.png, compare.md

Exemple :
  python compare_sessions.py data/session_A.txt:vide data/session_B.txt:marche data/session_C.txt:assis --name P1_P2_P3
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wisee_csi.parser import LLTF_SUBCARRIERS  # noqa: E402
from wisee_csi.metrics import subcarrier_quality, motion_index, correct_ht_scale_jumps  # noqa: E402
from analyze_session import load_session  # noqa: E402

COLORS = ["#7f7f7f", "#e6550d", "#31a354", "#3182bd", "#756bb1"]


def amplitudes(path, fs):
    frames = load_session(path)
    n = min(fr.csi.size for fr in frames)
    A_all, _ = correct_ht_scale_jumps(np.array([np.abs(fr.csi[:n]) for fr in frames]))
    A = A_all[:, LLTF_SUBCARRIERS:]                                                # HT-LTF corrige
    t = np.array([fr.meta["local_timestamp"] for fr in frames], dtype=float) / 1e6
    t = t - t[0]
    tg = np.arange(0, t[-1], 1.0 / fs)
    Ag = np.column_stack([np.interp(tg, t, A[:, j]) for j in range(A.shape[1])])
    return tg, Ag


def main():
    ap = argparse.ArgumentParser(description="Comparaison de sessions CSI WISEE")
    ap.add_argument("sessions", nargs="+", help="chemin:etiquette (la 1re = reference piece vide)")
    ap.add_argument("--name", default="comparaison")
    ap.add_argument("--fs", type=float, default=100.0)
    args = ap.parse_args()

    items = []
    for s in args.sessions:
        path, _, lab = s.rpartition(":") if ":" in os.path.basename(s) else (s, "", os.path.basename(s))
        items.append((path, lab))

    data = [(lab, *amplitudes(path, args.fs)) for path, lab in items]
    valid = subcarrier_quality(data[0][2])["valid"]                  # masque de la reference
    outdir = os.path.join("reports", args.name)
    os.makedirs(outdir, exist_ok=True)

    rows, series = [], []
    for lab, tg, Ag in data:
        mi = motion_index(Ag, valid, args.fs, win_s=1.0)
        series.append((lab, tg[:mi.size] + 0.5, mi))
        rows.append((lab, np.median(mi), np.percentile(mi, 90), mi.max()))

    ref = rows[0][1]
    fig, ax = plt.subplots(figsize=(11, 4))
    for (lab, tm, mi), c in zip(series, COLORS):
        ax.plot(tm, mi, color=c, lw=1, label=lab)
    ax.set_xlabel("temps (s)"); ax.set_ylabel("indice de mouvement")
    ax.set_title("Indice de mouvement (ecart-type glissant 1 s, %d sous-porteuses HT valides)" % valid.sum())
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "motion_compare.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    try:                                                   # matplotlib >= 3.9
        bp = ax.boxplot([s[2] for s in series], tick_labels=[s[0] for s in series], showfliers=False, patch_artist=True)
    except TypeError:                                      # matplotlib plus ancien
        bp = ax.boxplot([s[2] for s in series], labels=[s[0] for s in series], showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("indice de mouvement"); ax.set_title("Distribution de l'indice de mouvement")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "motion_boxplot.png"), dpi=130); plt.close(fig)

    lines = ["# Comparaison de sessions : %s" % args.name, "",
             "Sous-porteuses HT valides (choisies sur '%s') : %d / %d" % (rows[0][0], valid.sum(), valid.size), "",
             "| Session | Mediane | P90 | Max | Mediane / reference |", "|---|---|---|---|---|"]
    for lab, med, p90, mx in rows:
        lines.append("| %s | %.4f | %.4f | %.4f | x%.2f |" % (lab, med, p90, mx, med / ref))
    open(os.path.join(outdir, "compare.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nFigures dans : %s" % outdir)


if __name__ == "__main__":
    main()
