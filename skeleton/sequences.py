# -*- coding: utf-8 -*-
"""
Scénarios de test à plusieurs états successifs (ex. marche -> debout -> marche -> chute).

Deux sources de CSI :
  simule : le simulateur articulé rend une trajectoire CONTINUE avec de vraies transitions
           (s'arrêter, s'asseoir, se relever, tomber en marchant...) ; vérité terrain exacte.
  reel   : montage de segments de VRAIES prises mises de côté (jamais vues par le modèle de test,
           cf. holdout.py). Vrai signal ESP32, mais les raccords entre prises sont artificiels
           (la personne « change de place » d'un coup entre deux segments).

Un scénario s'écrit 'marche:4,debout:4,marche:4,chute:6' (état:durée en secondes).
'chute' = la chute elle-même puis la personne reste au sol jusqu'à la fin du segment.

    python -m skeleton.sequences "marche:4,assis:5,chute:6" --source simule
"""
import argparse
import json
import sys
import os

import numpy as np

from . import config as cfg
from . import labels as L
from . import poses as P
from . import synth
from .features import load_rx_csv, take_features

STATES = ["debout", "assis", "marche", "chute"]
PRESETS = {
    "marche → debout → marche → chute": "marche:4,debout:4,marche:4,chute:6",
    "marche → assis → chute": "marche:4,assis:5,chute:6",
    "debout → marche → assis → debout → marche": "debout:4,marche:5,assis:5,debout:4,marche:4",
    "assis → marche → debout → chute": "assis:5,marche:5,debout:4,chute:6",
    "marche → chute": "marche:5,chute:6",
    "debout → assis → debout → marche → chute": "debout:4,assis:5,debout:3,marche:4,chute:6",
}
G = {g: i for i, g in enumerate(cfg.GROUPS)}
POSE_HZ = 30.0


def parse_script(text):
    """'marche:4,chute:6' -> [('marche', 4.0), ('chute', 6.0)]"""
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        s, _, d = part.partition(":")
        if s not in STATES:
            raise ValueError("état inconnu : %r (choix : %s)" % (s, ", ".join(STATES)))
        d = float(d or 4)
        if not 1.0 <= d <= 30.0:
            raise ValueError("durée hors limites (1-30 s) : %s" % part)
        out.append((s, d))
    if not out:
        raise ValueError("scénario vide")
    if any(s == "chute" for s, _ in out[:-1]):
        raise ValueError("'chute' ne peut être que le dernier état (la personne reste au sol)")
    return out


def truth_segments(groups, fs):
    """Groupes par échantillon -> [(début_s, fin_s, groupe), ...]."""
    segs = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            segs.append((round(start / fs, 3), round(i / fs, 3), cfg.GROUPS[int(groups[start])]))
            start = i
    return segs


# --------------------------------------------------------------------------
# Source SIMULÉE : trajectoire continue avec transitions
# --------------------------------------------------------------------------
def sim_track(script, rng):
    """-> (kp_body, atom, pos, heading, info) à 90 Hz, comme synth.person_track."""
    H = rng.uniform(1.55, 1.95)
    dt = 1.0 / POSE_HZ
    stand, sit, ground = P.pose_stand(H, rng), P.pose_sit(H, rng), P.pose_ground(H, rng)
    pos = synth._free_spot(rng, 1.2)
    heading = rng.uniform(-np.pi, np.pi)
    phase = 0.0
    v = rng.uniform(0.9, 1.3)
    KP, AT, POS, HD = [], [], [], []
    prev_pose, prev_state = None, None
    m = 0.9                                                   # marge aux murs

    def emit(k, a):
        KP.append(k)
        AT.append(a)
        POS.append(pos.copy())
        HD.append(heading)

    for si, (state, dur) in enumerate(script):
        n = int(round(dur * POSE_HZ))
        # se lever avant de tomber si on était assis (compté comme 'debout')
        if state == "chute" and prev_state == "assis":
            for i in range(int(POSE_HZ)):
                w = (i + 1) / POSE_HZ
                emit((1 - w) * sit + w * stand, 1)
            prev_pose = stand
            n = max(n - int(POSE_HZ), int(POSE_HZ))
        trans = {("debout", "assis"): 1.0, ("assis", "debout"): 1.0, ("assis", "marche"): 1.2,
                 ("marche", "assis"): 1.2, ("marche", "debout"): 0.5, ("debout", "marche"): 0.5}
        tau = trans.get((prev_state, state), 0.0) if prev_pose is not None else 0.0
        t_fall = rng.uniform(0.6, 1.0)
        for i in range(n):
            t = i * dt
            if state == "debout":
                k, a = stand, 1
            elif state == "assis":
                k, a = sit, 2
            elif state == "marche":
                phase = (phase + 2 * np.pi * (v / 1.35) * dt) % (2 * np.pi)
                nxt = pos + v * dt * np.array([np.cos(heading), np.sin(heading)])
                if not (m < nxt[0] < cfg.ROOM_L - m and m < nxt[1] < cfg.ROOM_W - m):
                    heading += np.pi * rng.uniform(0.6, 1.0) * rng.choice([-1, 1])  # demi-tour
                else:
                    pos[:] = nxt
                ramp = min(1.0, t / 0.6) if prev_state != "marche" else 1.0
                k, a = P.pose_walk(phase, H, stride=0.3 + 0.7 * ramp), P.walk_atom(phase)
            else:                                             # chute : tombe puis reste au sol
                if t < t_fall:
                    al = t / t_fall
                    k, a = P.pose_fall(al, H), P.fall_atom(al)
                else:
                    k, a = ground, cfg.N_ATOMS - 1
            if tau and t < tau and state != "chute":
                w = t / tau
                k = (1 - w) * prev_pose + w * k
            emit(k, a)
        prev_pose, prev_state = KP[-1], state

    kp = np.array(KP)
    # respiration
    tt = np.arange(len(kp)) * dt
    kp[:, 5:7, 1] += 0.003 * np.sin(2 * np.pi * rng.uniform(0.2, 0.35) * tt)[:, None]
    # interpolation à la cadence des trames (90 Hz)
    n90 = int(len(kp) * cfg.TX_RATE_HZ / POSE_HZ)
    t90 = np.arange(n90) / cfg.TX_RATE_HZ
    flat = kp.reshape(len(kp), -1)
    kp90 = np.stack([np.interp(t90, tt, flat[:, c]) for c in range(flat.shape[1])], 1).reshape(n90, 17, 3)
    POS = np.array(POS)
    pos90 = np.stack([np.interp(t90, tt, POS[:, 0]), np.interp(t90, tt, POS[:, 1])], 1)
    hd90 = np.interp(t90, tt, np.unwrap(np.array(HD)))
    at90 = np.array(AT)[np.minimum((t90 * POSE_HZ).astype(int), len(AT) - 1)]
    return kp90, at90, pos90, hd90, {"height_m": round(H, 3), "speed_ms": round(v, 2)}


def simulate_sequence(script, seed=None, env_seed=None):
    rng = np.random.default_rng(seed)
    env_seed = int(rng.integers(50_000, 60_000)) if env_seed is None else env_seed   # séance jamais vue
    env = synth.Environment(env_seed)
    track = sim_track(script, rng)
    rx_out, lab, info = synth.simulate_take("sequence", rng, env, track=track)
    rx = {n: (ts, raw.astype(np.int16)) for n, (ts, raw, _ids) in rx_out.items()}
    groups = lab["group"]
    return {"rx": rx, "t0_ms": int(lab["t0_ms"]), "duration": len(groups) / cfg.TX_RATE_HZ,
            "truth": truth_segments(groups, cfg.TX_RATE_HZ),
            "truth_kp": lab["kp_body"][::int(cfg.TX_RATE_HZ // 10)],   # 10 Hz, pour l'affichage
            "meta": {"source": "simule", "seance_simulee": env_seed, **info}}


# --------------------------------------------------------------------------
# Source RÉELLE : montage de segments de vraies prises (mises de côté)
# --------------------------------------------------------------------------
_RAW_CACHE = {}


def _load_raw(take_dir):
    if take_dir in _RAW_CACHE:
        return _RAW_CACHE[take_dir]
    out = {}
    for name in cfg.RX_NAMES:
        p = os.path.join(take_dir, name + ".csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            a = np.loadtxt(p, delimiter=",", dtype=np.int64, ndmin=2)
            if len(a) >= 5:
                out[name] = (a[:, 0], a[:, 1:129].astype(np.int16))
    _RAW_CACHE[take_dir] = out
    return out


_FALL_CACHE = {}


def _fall_time(take_dir, raw, start):
    if take_dir not in _FALL_CACHE:
        X, _, _ = take_features({n: (ts, _raw_c(r)) for n, (ts, r) in raw.items()}, t_start_ms=start,
                                duration=cfg.TAKE_S)
        _FALL_CACHE[take_dir] = L.fall_time(X)[0]
    return _FALL_CACHE[take_dir]


def _raw_c(r):
    """raw (n,128) [im, re]... -> complexe (n,64) dans l'ordre du CSV."""
    r = r.astype(np.float32)
    return (r[:, 1::2] + 1j * r[:, 0::2])[:, cfg.SC_COLS]


def _c_raw(H64):
    raw = np.empty((len(H64), 128), np.int16)
    raw[:, 0::2] = np.clip(np.round(H64.imag), -127, 127)
    raw[:, 1::2] = np.clip(np.round(H64.real), -127, 127)
    return raw


def _blend(prev, ts, r, blend_s):
    """Raccord progressif : au début du segment, chaque sous-porteuse est ramenée au niveau moyen
    de la fin du segment précédent, puis la correction s'efface en blend_s secondes. La forme du canal
    change donc en douceur (comme un vrai changement de posture) au lieu de sauter d'un coup."""
    if prev is None or len(r) < 5 or blend_s <= 0:
        return r
    H = r.astype(np.float32)
    H = H[:, 1::2] + 1j * H[:, 0::2]                                    # (n,64)
    tt = (ts - ts[0]) / 1000.0
    cur = np.abs(H[tt < 1.0]).mean(0)
    ratio = np.where((cur > 0.5) & (prev > 0.5), prev / np.maximum(cur, 1e-3), 1.0)
    w = np.clip(1 - tt / blend_s, 0, 1)[:, None]
    return _c_raw(H * ratio[None, :] ** w)


def real_montage(script, takes_by_class, seed=None, blend_s=1.2):
    """takes_by_class = {'marche': [chemins de prises], ...} (prises jamais vues par le modèle).
    blend_s : durée du raccord progressif entre deux segments (0 = raccord brut)."""
    rng = np.random.default_rng(seed)
    t0 = 1794307330000
    cursor = 0.0
    rx = {n: ([], []) for n in cfg.RX_NAMES}
    truth, used = [], []
    prev_prof = {n: None for n in cfg.RX_NAMES}
    for state, dur in script:
        cands = takes_by_class.get(state, [])
        if not cands:
            raise ValueError("aucune prise réelle disponible pour '%s'" % state)
        take = cands[rng.integers(len(cands))]
        d = os.path.join(L.REAL_ROOT, take)
        raw = _load_raw(d)
        start = max(int(v[0][0]) for v in raw.values())
        end = min(int(v[0][-1]) for v in raw.values())
        avail = (end - start) / 1000.0
        if state == "chute":
            tf = _fall_time(d, raw, start)
            a = max(0.0, tf - 1.0)                            # 1 s debout avant la chute
            seg = min(dur, avail - a)
            # vérité : debout jusqu'à tf-0.4, chute jusqu'à tf+0.8, puis au sol
            for s0, s1, g in ((a, tf - L.FALL_BEFORE, "debout"), (tf - L.FALL_BEFORE, tf + L.FALL_AFTER, "chute"),
                              (tf + L.FALL_AFTER, a + seg, "au_sol")):
                s0, s1 = max(s0, a), min(s1, a + seg)
                if s1 > s0:
                    truth.append([round(cursor + s0 - a, 3), round(cursor + s1 - a, 3), g])
        else:
            seg = min(dur, avail - 0.1)
            a = rng.uniform(0, max(0.0, avail - seg))
            truth.append([round(cursor, 3), round(cursor + seg, 3), state])
        for n, (ts, r) in raw.items():
            sel = (ts >= start + a * 1000) & (ts < start + (a + seg) * 1000)
            rr = _blend(prev_prof[n], ts[sel], r[sel], blend_s)
            rx[n][0].append(t0 + np.round((cursor - a) * 1000).astype(np.int64) + (ts[sel] - start))
            rx[n][1].append(rr)
            if len(rr) >= 5:
                Hs = rr.astype(np.float32)
                tail = (ts[sel] - ts[sel][-1]) / 1000.0 > -1.0
                prev_prof[n] = np.abs(Hs[tail, 1::2] + 1j * Hs[tail, 0::2]).mean(0)
        used.append({"etat": state, "prise": take, "debut_s": round(a, 2), "duree_s": round(seg, 2)})
        cursor += seg
    out_rx = {n: (np.concatenate(t), np.concatenate(r)) for n, (t, r) in rx.items() if t and sum(map(len, t))}
    return {"rx": out_rx, "t0_ms": t0, "duration": cursor, "truth": truth, "truth_kp": None,
            "meta": {"source": "reel (montage de prises mises de côté, raccords progressifs %.1f s)" % blend_s,
                     "segments": used}}


def build(script_text, source="simule", seed=None, takes_by_class=None):
    script = parse_script(script_text)
    if source == "simule":
        return simulate_sequence(script, seed)
    if source == "reel":
        if takes_by_class is None:
            from .holdout import holdout_takes
            takes_by_class = holdout_takes()
        return real_montage(script, takes_by_class, seed)
    raise ValueError(source)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # console Windows (cp1252)
    ap = argparse.ArgumentParser()
    ap.add_argument("script", nargs="?", default=PRESETS["marche → debout → marche → chute"])
    ap.add_argument("--source", choices=["simule", "reel"], default="simule")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    s = build(a.script, a.source, a.seed)
    print(json.dumps({"duree_s": round(s["duration"], 2), "verite": s["truth"], "meta": s["meta"],
                      "trames": {n: len(v[0]) for n, v in s["rx"].items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
