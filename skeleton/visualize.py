# -*- coding: utf-8 -*-
"""Rendu matplotlib des squelettes (contrôle visuel, figures du rapport).

    python -m skeleton.visualize atoms                      -> skeleton/figures/atoms.png
    python -m skeleton.visualize pred <sortie_infer.json>   -> planche des poses prédites
"""
import json
import os
import sys

import numpy as np

from . import config as cfg
from .poses import SKELETON, atom_bank

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def draw_side(ax, kp, color="C0", title=None):
    """Vue de profil (y avant, z haut) + vue de face en transparence."""
    if not np.any(kp):
        ax.text(0.5, 0.5, "(personne absente)", ha="center", transform=ax.transAxes)
    for a, b in SKELETON:
        ax.plot([kp[a, 1], kp[b, 1]], [kp[a, 2], kp[b, 2]], "-", color=color, lw=2)
        ax.plot([kp[a, 0] - 1.2, kp[b, 0] - 1.2], [kp[a, 2], kp[b, 2]], "-", color="0.6", lw=1)
    ax.scatter(kp[:, 1], kp[:, 2], s=8, color=color, zorder=3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.1, 2.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def plot_atoms(path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = atom_bank()
    n = len(A)
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            draw_side(ax, A[i], title="%d %s" % (i, cfg.ATOM_GROUP[i]))
        else:
            ax.axis("off")
    fig.suptitle("Atomes de pose COCO-17 (gris : face, couleur : profil, avant = droite)")
    path = path or os.path.join(FIG_DIR, "atoms.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    return path


def plot_prediction(json_path, out=None, n=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(json_path, encoding="utf-8") as f:
        res = json.load(f)
    frames = res["frames"]
    sel = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    cols = 8
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows), squeeze=False)
    for ax, i in zip(axes.ravel(), sel):
        fr = frames[i]
        draw_side(ax, np.array(fr["keypoints_body"]),
                  color="C3" if fr["activity"] in ("chute", "au_sol") else "C0",
                  title="t=%.1fs %s (%.0f%%)" % (fr["t"], fr["activity"], 100 * fr["confidence"]))
    for ax in axes.ravel()[len(sel):]:
        ax.axis("off")
    fig.suptitle(os.path.basename(os.path.dirname(json_path)) + " / " + res.get("source", ""), fontsize=9)
    out = out or os.path.splitext(json_path)[0] + ".png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pred":
        print(plot_prediction(sys.argv[2]))
    else:
        print(plot_atoms())
