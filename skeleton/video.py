# -*- coding: utf-8 -*-
"""
Export vidéo d'une prise : squelette animé sur les 8 s + activité + frise des probabilités.

    python -m skeleton.video acquisition/data/DATASET/chute/p1/prise_010            -> .gif à côté de la prise
    python -m skeleton.video <prise> -o chute.mp4 --mode temps_reel                   (mp4 si ffmpeg est installé)

Mode par défaut : 'prise_entiere' (modèle qui voit les 8 s, comme une vidéo).
"""
import argparse
import os
import shutil

import numpy as np

from . import config as cfg
from .infer import CLIP_CKPT, DEFAULT_CKPT, infer_take, load_model
from .poses import SKELETON

LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
GROUP_COLORS = {"vide": "#9a9890", "debout": "#2a78d6", "assis": "#1baf7a", "marche": "#6250d6",
                "chute": "#e34948", "au_sol": "#008300"}


def project(kp, yaw=-0.7, pitch=0.22):
    """Projection perspective simple (même caméra que l'interface web)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    vx = kp[:, 0] * cy - kp[:, 1] * sy
    vy = kp[:, 0] * sy + kp[:, 1] * cy
    vz = kp[:, 2] - 0.9
    y2 = vz * cp - vy * sp
    d = vy * cp + vz * sp
    k = 4 / (4 + d)
    return np.stack([vx * k, (y2 + 0.9 * cp) * k], 1), d


def render(res, out_path, fps=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    frames = res["frames"]
    T = len(frames)
    fs = res["fs"]
    fps = fps or fs
    groups = cfg.GROUPS
    P = np.array([[f["probs"][g] for g in groups] for f in frames])
    t = np.array([f["t"] for f in frames])

    fig = plt.figure(figsize=(9, 6.2), dpi=90)
    ax = fig.add_axes([0.03, 0.30, 0.55, 0.64])
    axp = fig.add_axes([0.66, 0.38, 0.31, 0.50])
    axt = fig.add_axes([0.06, 0.06, 0.90, 0.17])
    fig.suptitle("%s   (mode : %s)" % (res["source"].split("data/")[-1], res.get("mode", "")), fontsize=10)

    # frise des probabilités (fixe) + curseur
    axt.stackplot(t, P.T, colors=[GROUP_COLORS[g] for g in groups], linewidth=0)
    axt.set_xlim(t[0], t[-1])
    axt.set_ylim(0, 1)
    axt.set_yticks([])
    axt.set_xlabel("temps (s)")
    cur = axt.axvline(t[0], color="k", lw=2)

    # éléments fixes, dessinés une seule fois (rendu ~10x plus rapide)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.25, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    for gx in np.linspace(-1.5, 1.5, 7):
        for a, b in (([gx, -1.5, 0], [gx, 1.5, 0]), ([-1.5, gx, 0], [1.5, gx, 0])):
            p, _ = project(np.array([a, b]))
            ax.plot(p[:, 0], p[:, 1], color="0.88", lw=0.8, zorder=0)
    bones = []
    for a, b in SKELETON:
        c = "#2563eb" if a in LEFT and b in LEFT else "#ea580c" if a in RIGHT and b in RIGHT else "#475569"
        bones.append(ax.plot([], [], color=c, lw=4, solid_capstyle="round")[0])
    joints = ax.scatter([], [], s=14, color="#334155", zorder=5)
    absent = ax.text(0, 1, "", ha="center", color="0.5")
    title = ax.text(-1.25, 1.95, "", fontsize=20, fontweight="bold", va="top")
    sub = ax.text(-1.25, 1.72, "", fontsize=10, va="top", color="0.3")
    bars = axp.barh(range(len(groups))[::-1], P[0], color=[GROUP_COLORS[x] for x in groups])
    axp.set_yticks(range(len(groups))[::-1])
    axp.set_yticklabels(groups, fontsize=9)
    axp.set_xlim(0, 1)
    axp.set_xticks([0, 0.5, 1])
    axp.set_title("probabilités", fontsize=9)

    def draw(i):
        f = frames[i]
        kp = np.array(f["keypoints_body"])
        visible = f["present"] and np.any(kp)
        if visible:
            p, d = project(kp)
            for (a, b), ln in zip(SKELETON, bones):
                ln.set_data(p[[a, b], 0], p[[a, b], 1])
                ln.set_zorder(4 - 0.1 * (d[a] + d[b]))          # le plus proche devant
            joints.set_offsets(p)
        for ln in bones:
            ln.set_visible(visible)
        joints.set_visible(visible)
        absent.set_text("" if visible else "personne absente")
        g = f["activity"]
        title.set_text(g.upper())
        title.set_color(GROUP_COLORS[g])
        sub.set_text("%d %%   t = %.2f s" % (100 * f["confidence"], f["t"]))
        for bar, v in zip(bars, P[i]):
            bar.set_width(v)
        cur.set_xdata([f["t"], f["t"]])
        return []

    anim = animation.FuncAnimation(fig, draw, frames=T, interval=1000 / fps)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4" and shutil.which("ffmpeg"):
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    else:
        if ext == ".mp4":
            out_path = os.path.splitext(out_path)[0] + ".gif"
            print("ffmpeg introuvable : export en GIF ->", out_path)
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--mode", choices=["prise_entiere", "temps_reel"], default="prise_entiere")
    a = ap.parse_args()
    ck = CLIP_CKPT if a.mode == "prise_entiere" and os.path.exists(CLIP_CKPT) else DEFAULT_CKPT
    res = infer_take(a.take_dir, load_model(ck))
    out = a.out or os.path.join(a.take_dir, "skeleton_%s.gif" % res["mode"])
    print(render(res, out))


if __name__ == "__main__":
    main()
