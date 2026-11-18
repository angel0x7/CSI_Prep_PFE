# -*- coding: utf-8 -*-
"""
Simulateur CSI "corps articulé" pour générer des prises synthétiques AVEC vérité terrain squelette.

Différence avec acquisition/python/wisee_csi/channel_model.py (1 trajet humain abstrait) :
ici la personne est un ensemble de 36 réflecteurs (17 keypoints COCO + milieux des os)
qui bougent selon le squelette, dans une salle de classe avec murs, tables et l'opérateur
assis au PC (présent pendant toutes les vraies prises). On reproduit aussi le matériel :
réponse en fréquence propre à chaque RX, AGC, bruit, quantification int8, pertes de trames
en rafales calibrées sur l'index.csv réel, horodatage ms par récepteur.

Canal (par RX, par trame, par sous-porteuse f) :
  H(f) = g_rx(f) * [ LOS * ombrage(corps)  +  Σ murs  +  Σ tables/chaises
                     + Σ_{réflecteurs m} σ w_m / (d1 d2) exp(-j2π(fc+f)(d1+d2)/c) ]

Usage (écrit des prises au format DATASET : RXn.csv + summary.json + labels.npz) :
  python -m skeleton.synth --n-per-class 40 --out acquisition/data/DATASET_SYNTH
"""
import argparse
import json
import os

import numpy as np

from . import config as cfg
from . import poses as P

FREQS = cfg.FC + cfg.SC_FREQ_IDX * cfg.DF                    # (52,) Hz
K = 2 * np.pi * FREQS / cfg.C                                # nombre d'onde par sous-porteuse
ACTIVITIES = ["vide", "debout", "assis", "marche", "chute"]

# poids de réflexion de chaque keypoint COCO (tronc fort, extrémités faibles)
KP_W = np.array([0.05, 0.02, 0.02, 0.03, 0.03, 0.30, 0.30, 0.10, 0.10, 0.05, 0.05,
                 0.30, 0.30, 0.12, 0.12, 0.05, 0.05])
BONE_W = {(5, 11): 0.45, (6, 12): 0.45, (5, 6): 0.35, (11, 12): 0.35}
BONES = [b for b in P.SKELETON if b[0] >= 5 and b[1] >= 5]  # os du corps (sans le visage)
PT_W = np.concatenate([KP_W, [BONE_W.get(b, BONE_W.get(b[::-1], 0.12)) for b in BONES]])

_REAL_LOSS = None


def _real_loss_rates():
    """Distribution empirique des pertes par RX (index.csv du vrai DATASET)."""
    global _REAL_LOSS
    if _REAL_LOSS is None:
        path = os.path.join(os.path.dirname(__file__), "..", "acquisition", "data", "DATASET", "index.csv")
        vals = []
        try:
            import csv
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    for k in range(1, 5):
                        v = r.get("pertes_RX%d_pct" % k, "")
                        if v not in ("", None):
                            vals.append(float(v) / 100)
        except OSError:
            pass
        _REAL_LOSS = np.clip(np.array(vals or [0.45]), 0.0, 0.85)
    return _REAL_LOSS


# --------------------------------------------------------------------------
# Environnement statique
# --------------------------------------------------------------------------
class Environment(object):
    """Une 'séance' : mobilier, coefficients des murs, réponse matérielle des RX, opérateur."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.tx = np.array(cfg.TX_POS)
        self.rx = {n: np.array(p) for n, p in cfg.RX_POS.items()}
        # tables d'élèves : 5 rangées x 6 places, plateau à 0.75 m + chaises à 0.45 m
        xs = np.linspace(1.8, cfg.ROOM_L - 0.8, 5)
        ys = np.linspace(0.9, cfg.ROOM_W - 0.9, 6)
        desks = [(x + rng.normal(0, 0.1), y + rng.normal(0, 0.1), 0.75) for x in xs for y in ys]
        chairs = [(x - 0.45 + rng.normal(0, 0.1), y + rng.normal(0, 0.1), 0.45) for x, y, _ in desks]
        self.scat = np.array(desks + chairs)
        self.scat_s = rng.uniform(0.02, 0.10, len(self.scat))
        self.wall_g = rng.uniform(0.25, 0.6, 6)              # x=0, x=L, y=0, y=W, sol, plafond
        # réponse en fréquence de chaque récepteur (filtre + antenne), lisse
        self.ripple = {}
        for n in self.rx:
            c = rng.normal(0, 1, 6)
            x = np.linspace(-1, 1, cfg.N_SC)
            self.ripple[n] = 1 + 0.2 * sum(c[i] * np.cos((i + 1) * np.pi * x / 2 + c[-1]) for i in range(4))
        # opérateur assis au PC (fixe pour la séance)
        self.operator_pos = np.array([rng.uniform(0.8, 2.0), rng.uniform(5.2, 6.3)])
        self.operator_heading = rng.uniform(-np.pi, np.pi)
        self.static = {n: self._static(n) for n in self.rx}

    def _static(self, name):
        rx, tx = self.rx[name], self.tx
        H = np.exp(-1j * K * np.linalg.norm(rx - tx)) / np.linalg.norm(rx - tx)
        dims = [(0, 0.0), (0, cfg.ROOM_L), (1, 0.0), (1, cfg.ROOM_W), (2, 0.0), (2, cfg.ROOM_H)]
        for g, (ax, pos) in zip(self.wall_g, dims):           # sources images du 1er ordre
            img = tx.copy()
            img[ax] = 2 * pos - img[ax]
            d = np.linalg.norm(rx - img)
            H = H - g * np.exp(-1j * K * d) / d
        d1 = np.linalg.norm(self.scat - tx, axis=1)
        d2 = np.linalg.norm(self.scat - rx, axis=1)
        H = H + (self.scat_s / (d1 * d2))[:, None].T @ np.exp(-1j * np.outer(d1 + d2, K))
        return H.ravel()


# --------------------------------------------------------------------------
# Trajectoires / poses
# --------------------------------------------------------------------------
def _smooth_noise(rng, n, std, alpha=0.97):
    x = np.zeros(n)
    e = rng.normal(0, std * np.sqrt(1 - alpha ** 2), n)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + e[i]
    return x


def _free_spot(rng, margin=0.8):
    return np.array([rng.uniform(margin, cfg.ROOM_L - margin), rng.uniform(margin, cfg.ROOM_W - margin)])


def person_track(activity, rng, n, dt):
    """Retourne kp_body (n,17,3), atom (n,), pos (n,2), heading (n,), info dict.
    Poses calculées à ~30 Hz puis interpolées (rapide)."""
    H = rng.uniform(1.55, 1.95)
    step = 3                                                  # 1 pose sur 3 trames
    idx = np.arange(0, n + step, step)
    m = len(idx)
    t = idx * dt
    pos = np.tile(_free_spot(rng), (m, 1))
    heading = np.full(m, rng.uniform(-np.pi, np.pi))
    atom = np.zeros(m, dtype=int)
    kps = np.zeros((m, 17, 3))
    info = {"height_m": round(H, 3)}

    if activity in ("debout", "assis"):
        base = P.pose_stand(H, rng) if activity == "debout" else P.pose_sit(H, rng)
        atom[:] = 1 if activity == "debout" else 2
        kps[:] = base
        pos += np.stack([_smooth_noise(rng, m, 0.006, 0.99), _smooth_noise(rng, m, 0.006, 0.99)], 1)
        # petits gestes (bras / tête) de temps en temps
        for _ in range(rng.poisson(0.8)):
            c, w = rng.integers(0, m), rng.integers(10, 40)
            env = np.exp(-0.5 * ((np.arange(m) - c) / (w / 3)) ** 2)
            j = rng.choice([7, 9, 8, 10, 0])
            amp = rng.normal(0, 0.05, 3)
            kps[:, j] += env[:, None] * amp
            if j in (7, 8):
                kps[:, j + 2] += env[:, None] * amp * 1.8
    elif activity == "marche":
        v = rng.uniform(0.8, 1.4)
        a = _free_spot(rng, 1.0)
        L = rng.uniform(3.0, 6.0)
        for _ in range(50):                                   # segment aller-retour dans la salle
            th = rng.uniform(-np.pi, np.pi)
            b = a + L * np.array([np.cos(th), np.sin(th)])
            if 0.8 < b[0] < cfg.ROOM_L - 0.8 and 0.8 < b[1] < cfg.ROOM_W - 0.8:
                break
            L *= 0.9
        else:
            b = a
        s = rng.uniform(0, L)                                 # abscisse curviligne de départ
        direction = rng.choice([-1.0, 1.0])
        phase = rng.uniform(0, 2 * np.pi)
        turn_left = 0.0
        u = (b - a) / max(L, 1e-6)
        th_ab = np.arctan2(u[1], u[0])
        h = th_ab if direction > 0 else th_ab + np.pi
        stride = v / 1.15
        dts = step * dt
        for i in range(m):
            if turn_left > 0:                                 # demi-tour sur place ~0.8 s
                turn_left -= dts
                h += np.pi * dts / 0.8 * rng.choice([1, 1])
                phase += 2 * np.pi * 0.9 * dts
                cur_stride = 0.4
            else:
                s += direction * v * dts
                if s > L or s < 0:
                    s = np.clip(s, 0, L)
                    direction *= -1
                    turn_left = 0.8
                phase += 2 * np.pi * (v / 1.35) * dts
                cur_stride = stride
                h = th_ab if direction > 0 else th_ab + np.pi
            pos[i] = a + u * s
            heading[i] = h
            kps[i] = P.pose_walk(phase % (2 * np.pi), H, cur_stride)
            atom[i] = P.walk_atom(phase % (2 * np.pi))
        info.update(speed_ms=round(v, 2))
    elif activity == "chute":
        t_f = rng.uniform(1.0, 3.5)                           # début de la chute
        dur = rng.uniform(0.6, 1.2)
        stand = P.pose_stand(H, rng)
        ground = P.pose_ground(H, rng)
        for i in range(m):
            if t[i] < t_f:
                kps[i], atom[i] = stand, 1
            elif t[i] < t_f + dur:
                al = (t[i] - t_f) / dur
                kps[i], atom[i] = P.pose_fall(al, H), P.fall_atom(al)
            else:
                kps[i], atom[i] = ground, cfg.N_ATOMS - 1
        pos += np.stack([_smooth_noise(rng, m, 0.01), _smooth_noise(rng, m, 0.01)], 1)
        info.update(fall_start_s=round(t_f, 3), fall_duration_s=round(dur, 3))
    elif activity == "vide":
        atom[:] = 0
    else:
        raise ValueError(activity)

    # respiration : poitrine (épaules) d'avant en arrière, ~3 mm
    if activity != "vide":
        fb = rng.uniform(0.2, 0.35)
        br = 0.003 * np.sin(2 * np.pi * fb * t + rng.uniform(0, 6.3))
        kps[:, 5:7, 1] += br[:, None]
        info["breathing_hz"] = round(fb, 3)

    # interpolation à la cadence des trames
    tt = np.arange(n) * dt
    out_k = np.empty((n, 17, 3))
    flat = kps.reshape(m, -1)
    for c in range(flat.shape[1]):
        out_k.reshape(n, -1)[:, c] = np.interp(tt, t, flat[:, c])
    pos_i = np.stack([np.interp(tt, t, pos[:, 0]), np.interp(tt, t, pos[:, 1])], 1)
    head_i = np.interp(tt, t, np.unwrap(heading))
    atom_i = atom[np.minimum(np.round(tt / (step * dt)).astype(int), m - 1)]
    return out_k, atom_i, pos_i, head_i, info


def body_points(kp_world):
    """(n,17,3) -> (n,36,3) : keypoints + milieux des os."""
    mids = np.stack([(kp_world[:, a] + kp_world[:, b]) / 2 for a, b in BONES], 1)
    return np.concatenate([kp_world, mids], 1)


def _body_channel(pts, w, tx, rx, sigma):
    d1 = np.linalg.norm(pts - tx, axis=-1)                    # (n,M)
    d2 = np.linalg.norm(pts - rx, axis=-1)
    a = sigma * w[None, :] / (d1 * d2)
    return np.einsum("nm,nmf->nf", a, np.exp(-1j * (d1 + d2)[..., None] * K[None, None, :]))


def _shadow(pts, tx, rx):
    """Atténuation du trajet direct quand le corps coupe la ligne S-RX (zone de Fresnel)."""
    v = rx - tx
    L = np.linalg.norm(v)
    s = np.clip(((pts - tx) @ v) / L ** 2, 0, 1)              # (n,M)
    proj = tx + s[..., None] * v
    dist = np.linalg.norm(pts - proj, axis=-1)
    lam = cfg.C / cfg.FC
    rf = np.sqrt(lam * s * (1 - s) * L + 1e-6)
    dmin = (dist / (rf + 0.15)).min(axis=1)
    return 1 - 0.65 * np.exp(-dmin ** 2)


# --------------------------------------------------------------------------
# Matériel
# --------------------------------------------------------------------------
def _loss_mask(rng, n, p, burst):
    """Modèle de Gilbert-Elliott : taux de perte p, longueur moyenne des rafales burst."""
    if p <= 0:
        return np.ones(n, bool)
    q_bg = 1.0 / burst                                        # sortie de l'état 'perdu'
    q_gb = p * q_bg / max(1 - p, 1e-3)
    keep = np.empty(n, bool)
    bad = rng.random() < p
    r = rng.random(n)
    for i in range(n):
        bad = (r[i] >= q_bg) if bad else (r[i] < q_gb)
        keep[i] = not bad
    return keep


def _quantize(H, rng, level, noise, agc_beta=0.5, agc_jitter=0.03):
    """AGC partielle (niveau moyen 'level' LSB ; une fraction agc_beta des variations de puissance
    reste visible, comme sur le vrai ESP32) + bruit + rotation de phase aléatoire + int8."""
    n = H.shape[0]
    m = np.abs(H).mean(1, keepdims=True)
    r = m / m.mean()
    drift = np.exp(_smooth_noise(rng, n, agc_jitter, 0.995)[:, None])
    g = level / m * r ** agc_beta * drift * np.exp(rng.normal(0, agc_jitter, (n, 1)))
    Hq = H * g
    ph = rng.uniform(-np.pi, np.pi, (n, 1)) + rng.normal(0, 0.05, (n, 1)) * cfg.SC_FREQ_IDX[None, :]
    Hq = Hq * np.exp(1j * ph)
    Hq = Hq + noise * (rng.standard_normal(Hq.shape) + 1j * rng.standard_normal(Hq.shape))
    full = np.zeros((n, 64), complex)
    full[:, cfg.SC_COLS] = Hq
    raw = np.empty((n, 128), np.int16)
    raw[:, 0::2] = np.clip(np.round(full.imag), -127, 127)
    raw[:, 1::2] = np.clip(np.round(full.real), -127, 127)
    return raw


# --------------------------------------------------------------------------
# Une prise
# --------------------------------------------------------------------------
def simulate_take(activity, rng, env, duration=cfg.TAKE_S, sigma=None, loss=True, ideal=False, track=None):
    """ideal=True : acquisition parfaite (aucune perte, bruit et AGC minimes) -- sert à séparer,
    dans l'étude de faisabilité, les limites du DATASET de celles du MATÉRIEL / de la physique.
    track = (kp_body, atom, pos, heading, info) déjà calculé (ex. scénario à plusieurs états,
    cf. sequences.py) ; sinon une trajectoire à activité unique est tirée."""
    if ideal:
        loss = False
    dt = 1.0 / cfg.TX_RATE_HZ
    if track is None:
        n = int(duration * cfg.TX_RATE_HZ)
        kp_body, atom, pos, heading, info = person_track(activity, rng, n, dt)
    else:
        kp_body, atom, pos, heading, info = track
        n = len(kp_body)
    present = activity != "vide"
    sigma = rng.uniform(0.6, 1.4) * SIGMA_BODY if sigma is None else sigma

    pts = []
    if present:
        kp_w = np.stack([P.to_world(k, x, y, h) for k, (x, y), h in zip(kp_body, pos, heading)])
        pts.append((body_points(kp_w), PT_W))
    # opérateur assis au PC, avec petits mouvements (frappe, souris)
    op = P.to_world(P.pose_sit(), env.operator_pos[0], env.operator_pos[1], env.operator_heading)
    op_t = np.repeat(op[None], n, 0)
    op_t[:, 9:11] += _smooth_noise(rng, n, 0.02, 0.9)[:, None, None] * np.array([1, 1, 0.3])
    op_t[:, :5] += _smooth_noise(rng, n, 0.01, 0.98)[:, None, None]
    pts.append((body_points(op_t), PT_W))

    t0 = 1794307330000 + int(rng.integers(0, 10 ** 9))
    rx_out = {}
    for name, rx in env.rx.items():
        Hs = env.static[name][None, :] * np.ones((n, 1))
        if present:
            Hs = Hs - (1 - _shadow(pts[0][0], env.tx, rx))[:, None] * np.exp(
                -1j * K * np.linalg.norm(rx - env.tx))[None, :] / np.linalg.norm(rx - env.tx)
        Hh = sum(_body_channel(p, w, env.tx, rx, sigma) for p, w in pts)
        Hc = (Hs + Hh) * env.ripple[name][None, :]
        level = np.exp(rng.uniform(np.log(5), np.log(32)))
        if ideal:
            raw = _quantize(Hc, rng, 30.0, noise=0.1, agc_beta=0.0, agc_jitter=0.002)
        else:
            raw = _quantize(Hc, rng, level, noise=rng.uniform(0.35, 0.8),
                            agc_beta=rng.uniform(0.15, 0.5), agc_jitter=rng.uniform(0.01, 0.03))
        keep = _loss_mask(rng, n, rng.choice(_real_loss_rates()), rng.uniform(2, 6)) if loss \
            else np.ones(n, bool)
        ts = t0 + np.round(np.arange(n) * dt * 1000 + rng.normal(0, 1.0, n) + rng.normal(0, 4)).astype(np.int64)
        rx_out[name] = (ts[keep], raw[keep], np.nonzero(keep)[0])

    labels = dict(t=np.arange(n) * dt, t0_ms=t0, kp_body=kp_body.astype(np.float32),
                  atom=atom.astype(np.int16), group=cfg.ATOM_GROUP_IDX[atom].astype(np.int16),
                  pos=pos.astype(np.float32), heading=heading.astype(np.float32))
    info.update(activity=activity, sigma=round(float(sigma), 4))
    return rx_out, labels, info


SIGMA_BODY = 0.6           # réflectivité du corps (calibrée : voir calibrate.py)


# --------------------------------------------------------------------------
# Écriture au format DATASET
# --------------------------------------------------------------------------
def write_take(d, rx_out, labels, info):
    os.makedirs(d, exist_ok=True)
    summ = {"source": "skeleton.synth (SIMULATION)", "dir": d.replace("\\", "/"), "note": "synthétique", "receivers": {}}
    for name, (ts, raw, ids) in rx_out.items():
        arr = np.concatenate([ts[:, None], raw.astype(np.int64)], 1)
        np.savetxt(os.path.join(d, name + ".csv"), arr, fmt="%d", delimiter=",")
        n_emit = len(labels["t"])
        summ["receivers"][name] = {"frames": int(len(ts)), "duration_s": cfg.TAKE_S,
                                   "mean_fps": round(len(ts) / cfg.TAKE_S, 1),
                                   "lost": int(n_emit - len(ts)), "loss_pct": round(100 * (1 - len(ts) / n_emit), 2)}
    summ["simulation"] = info
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(d, "labels.npz"), **labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("acquisition", "data", "DATASET_SYNTH"))
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--n-env", type=int, default=8, help="nombre de 'séances' (mobilier, matériel) différentes")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    envs = [Environment(a.seed * 1000 + i) for i in range(a.n_env)]
    rows = ["chemin,activite,personnes,prise,environnement,pertes_max_pct,info"]
    for act in ACTIVITIES:
        p = 0 if act == "vide" else 1
        for i in range(a.n_per_class):
            e = int(rng.integers(len(envs)))
            rx_out, labels, info = simulate_take(act, rng, envs[e])
            rel = "%s/p%d/prise_%03d" % (act, p, i + 1)
            write_take(os.path.join(a.out, rel), rx_out, labels, info)
            lost = max(100 * (1 - len(v[0]) / len(labels["t"])) for v in rx_out.values())
            rows.append('%s,%s,%d,prise_%03d,%d,%.1f,"%s"' % (rel, act, p, i + 1, e, lost,
                                                         json.dumps(info).replace('"', "'")))
        print(act, "ok")
    with open(os.path.join(a.out, "index.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
