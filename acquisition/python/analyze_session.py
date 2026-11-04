# -*- coding: utf-8 -*-
"""
WISEE - Analyse hors ligne d'une session CSI enregistree.

Produit dans reports/<session>/ :
  quality.png    qualite par sous-porteuse (lesquelles masquer)
  heatmap.png    amplitude (temps x sous-porteuses valides)
  motion.png     indicateur de mouvement dans le temps (+ verite terrain si fournie)
  timing.png     trames/s et intervalles entre trames (regularite du flux)
  breathing.png  spectre basse frequence sur les phases immobiles (respiration)
  report.md      chiffres cles, reutilisables dans le plan de validation

Exemples :
  python analyze_session.py data/session_20261007_153000.txt --labels data/sim_labels.csv
  python analyze_session.py data/session_20261007_153000.txt --part lltf
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wisee_csi.parser import parse_line, LLTF_SUBCARRIERS  # noqa: E402
from wisee_csi.metrics import StreamStats, subcarrier_quality, motion_index, correct_ht_scale_jumps  # noqa: E402

COLORS = {"empty": "#d9d9d9", "stand": "#9ecae1", "sit": "#a1d99b", "walk": "#fdae6b",
          "fall": "#de2d26", "lying": "#bcbddc"}
STATIC = ("stand", "sit", "lying")


def load_session(path):
    frames = []
    with open(path) as f:
        for line in f:
            ts, _, body = line.partition("\t")
            fr = parse_line(body, host_ts=float(ts)) if body else None
            if fr is not None:
                frames.append(fr)
    if not frames:
        raise SystemExit("Aucune trame CSI valide dans %s" % path)
    return frames


def load_labels(path):
    segs = []
    if path and os.path.exists(path):
        with open(path) as f:
            next(f)
            for line in f:
                a, b, lab = line.strip().split(",")
                segs.append((float(a), float(b), lab))
    return segs


def shade(ax, segs):
    seen = set()
    for a, b, lab in segs:
        ax.axvspan(a, b, color=COLORS.get(lab, "#eeeeee"), alpha=0.45, lw=0,
                   label=None if lab in seen else lab)
        seen.add(lab)


def breathing_rate(x, fs, fmin=0.1, fmax=0.6):
    """Frequence respiratoire (respirations/min) = pic du spectre entre fmin et fmax Hz."""
    x = x - np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size))   # detrend
    win = np.hanning(x.size)
    nfft = 8 * x.size                    # zero-padding : interpolation du spectre (resolution affichee)
    spec = np.abs(np.fft.rfft(x * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return None, freqs, spec
    f_peak = freqs[band][np.argmax(spec[band])]
    return 60.0 * f_peak, freqs, spec


def main():
    ap = argparse.ArgumentParser(description="Analyse d'une session CSI WISEE")
    ap.add_argument("session")
    ap.add_argument("--labels", default=None, help="verite terrain CSV (start_s,end_s,label)")
    ap.add_argument("--part", choices=["lltf", "ht", "all"], default="ht",
                    help="lltf = 64 premieres sous-porteuses ; ht = HT-LTF (apres le LLTF)")
    ap.add_argument("--fs", type=float, default=100.0, help="frequence de reechantillonnage (Hz)")
    ap.add_argument("--time", choices=["device", "host"], default="device",
                    help="horloge : local_timestamp de l'ESP32 (defaut) ou horodatage PC")
    args = ap.parse_args()

    frames = load_session(args.session)
    name = os.path.splitext(os.path.basename(args.session))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(args.session)))
    if parent.startswith("multi_"):                       # session multi-recepteurs : multi_xxx_RX1
        name = parent + "_" + name
    outdir = os.path.join("reports", name)
    os.makedirs(outdir, exist_ok=True)

    # --- flux : regularite, pertes ------------------------------------------------
    stats = StreamStats()
    for fr in frames:
        stats.update(fr)
    summ = stats.summary()

    # --- matrice d'amplitude ----------------------------------------------------------
    n = min(fr.csi.size for fr in frames)
    sl = {"lltf": slice(0, LLTF_SUBCARRIERS), "ht": slice(LLTF_SUBCARRIERS, n), "all": slice(0, n)}[args.part]
    if sl.start >= n:
        sl = slice(0, n)                                   # trames sans HT-LTF
    A_all = np.array([np.abs(fr.csi[:n]) for fr in frames])
    A_all, frac_fix = correct_ht_scale_jumps(A_all)        # sauts x2 de la partie HT-LTF (artefact ESP32-S3)
    print("Trames HT corrigees (saut x2 / /2) : %.1f %%" % (100 * frac_fix))
    A = A_all[:, sl]
    if args.time == "device" and isinstance(frames[0].meta.get("local_timestamp"), int):
        t = np.array([fr.meta["local_timestamp"] for fr in frames], dtype=float) / 1e6
    else:
        t = np.array([fr.host_ts for fr in frames])
    t = t - t[0]

    # reechantillonnage sur une grille reguliere (le flux reel n'est pas regulier)
    tg = np.arange(0, t[-1], 1.0 / args.fs)
    Ag = np.column_stack([np.interp(tg, t, A[:, j]) for j in range(A.shape[1])])

    segs = load_labels(args.labels)
    quiet = Ag
    if segs:                                                # qualite mesuree sur la piece vide
        m = np.zeros(tg.size, bool)
        for a, b, lab in segs:
            if lab == "empty":
                m |= (tg >= a) & (tg < b)
        if m.sum() > args.fs * 2:
            quiet = Ag[m]
    q = subcarrier_quality(quiet)
    valid = q["valid"]
    mi = motion_index(Ag, valid, args.fs, win_s=1.0)
    tm = tg[:mi.size] + 0.5

    # --- figures ----------------------------------------------------------------------
    k = np.arange(A.shape[1])
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].bar(k, q["mean"], color=np.where(valid, "#0E7C72", "#de2d26"))
    ax[0].set_ylabel("amplitude moyenne")
    ax[0].set_title("Qualite par sous-porteuse (%s) : %d valides / %d  (rouge = a masquer)"
                    % (args.part, valid.sum(), valid.size))
    ax[1].bar(k, np.minimum(q["cv"], 1.0), color=np.where(valid, "#0E7C72", "#de2d26"))
    ax[1].set_ylabel("coef. de variation")
    ax[1].set_xlabel("indice de sous-porteuse (ordre du buffer ESP32)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "quality.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(Ag[:, valid].T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, tg[-1], 0, valid.sum()])
    ax.set_xlabel("temps (s)"); ax.set_ylabel("sous-porteuses valides")
    ax.set_title("Amplitude CSI dans le temps")
    fig.colorbar(im, ax=ax, label="amplitude")
    for a, b, lab in segs:
        ax.axvline(a, color="w", lw=0.8, ls="--")
        ax.text((a + b) / 2, valid.sum() * 0.98, lab, color="w", ha="center", va="top", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "heatmap.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    shade(ax, segs)
    ax.plot(tm, mi, color="#16202B", lw=1)
    ax.set_xlabel("temps (s)"); ax.set_ylabel("indice de mouvement")
    ax.set_title("Indicateur de mouvement (ecart-type glissant 1 s, sous-porteuses valides)")
    if segs:
        ax.legend(loc="upper right", fontsize=8, ncol=6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "motion.png"), dpi=130); plt.close(fig)

    th = np.array([fr.host_ts for fr in frames])      # horloge PC : c'est la que se voit le jitter
    th = th - th[0]
    per_s = np.bincount(np.floor(th).astype(int))[:-1] if th[-1] >= 1 else np.array([len(th)])
    iv = np.diff(th) * 1000
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(per_s, color="#0E7C72"); ax[0].set_xlabel("seconde"); ax[0].set_ylabel("trames recues")
    ax[0].set_title("Trames par seconde")
    ax[1].hist(iv, bins=60, color="#0E7C72"); ax[1].set_xlabel("intervalle entre trames (ms)")
    ax[1].set_yscale("log"); ax[1].set_title("Regularite du flux (horloge PC)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "timing.png"), dpi=130); plt.close(fig)

    # --- respiration sur les phases immobiles -----------------------------------------
    # 1re composante principale des amplitudes normalisees (la plus sensible au mouvement)
    Z = Ag[:, valid] / np.maximum(Ag[:, valid].mean(axis=0), 1e-6)
    Z = Z - Z.mean(axis=0)
    breath = []
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for a, b, lab in (segs if segs else [(0.0, tg[-1], "session")]):
        if segs and lab not in STATIC:
            continue
        m = (tg >= a + 1) & (tg < b - 1)
        if m.sum() < args.fs * 15:                       # au moins 15 s pour une estimation
            continue
        _, _, vt = np.linalg.svd(Z[m], full_matrices=False)
        pc = Z[m] @ vt[0]
        bpm, freqs, spec = breathing_rate(pc, args.fs)
        breath.append((lab, a, b, bpm))
        sel = freqs <= 1.0
        ax.plot(freqs[sel], spec[sel] / spec[sel].max(), label="%s (%.0f-%.0f s) : %.1f resp/min" % (lab, a, b, bpm))
    ax.axvspan(0.1, 0.6, color="#eeeeee", zorder=0)
    ax.set_xlabel("frequence (Hz)"); ax.set_ylabel("puissance normalisee")
    ax.set_title("Spectre basse frequence (bande respiratoire 0.1-0.6 Hz en gris)")
    if breath:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "breathing.png"), dpi=130); plt.close(fig)

    # --- rapport ----------------------------------------------------------------------
    lines = ["# Rapport de session CSI : %s" % name, "",
             "## Flux", "", "| Indicateur | Valeur |", "|---|---|"]
    for key in ("frames", "duration_s", "mean_fps", "lost", "loss_pct", "duplicates",
                "bad_lines", "interval_mean_ms", "interval_p95_ms", "interval_max_ms"):
        lines.append("| %s | %s |" % (key, summ[key]))
    lines += ["", "## Sous-porteuses (%s)" % args.part, "",
              "%d valides sur %d. Indices a masquer : %s" %
              (valid.sum(), valid.size, ", ".join(str(i) for i in np.where(~valid)[0]))]
    if segs:
        lines += ["", "## Indice de mouvement par activite (verite terrain)", "",
                  "| Activite | Debut (s) | Fin (s) | Indice moyen | Indice max |", "|---|---|---|---|---|"]
        for a, b, lab in segs:
            m = (tm >= a + 0.5) & (tm < b - 0.5)
            if m.any():
                lines.append("| %s | %.0f | %.0f | %.4f | %.4f |" % (lab, a, b, mi[m].mean(), mi[m].max()))
    if breath:
        lines += ["", "## Respiration estimee (phases immobiles)", "",
                  "| Segment | Debut (s) | Fin (s) | Respirations/min |", "|---|---|---|---|"]
        for lab, a, b, bpm in breath:
            lines.append("| %s | %.0f | %.0f | %.1f |" % (lab, a, b, bpm))
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("\n".join(lines))
    print("\nFigures et rapport dans : %s" % outdir)


if __name__ == "__main__":
    main()
