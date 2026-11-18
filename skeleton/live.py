# -*- coding: utf-8 -*-
"""
Classification EN DIRECT de l'état (debout / assis / marche / chute / au sol / vide) à partir du CSI.

Le même moteur sert :
  - au banc de test : rejeu d'un scénario (sequences.py) au rythme réel, comme si les trames
    arrivaient des ESP32 (utilisé par l'interface, onglet « Test en direct ») ;
  - au vrai système : écoute UDP des récepteurs ESP32 (même protocole que record_multi.py).

    python -m skeleton.live --scenario "marche:4,debout:4,marche:4,chute:6" --source simule
    python -m skeleton.live --udp 5005 --map 192.168.4.201=RX1 192.168.4.202=RX2 192.168.4.203=RX3 192.168.4.204=RX4

Toutes les 100 ms : features des 6 dernières secondes -> modèle CAUSAL (passé uniquement) ->
probabilités lissées -> machine à états (un nouvel état est validé dès que sa probabilité lissée dépasse 0.55,
sauf la chute, signalée immédiatement) -> squelette animé.
"""
import argparse
import collections
import os
import sys
import threading
import time

import numpy as np

from . import config as cfg
from .features import raw_to_complex, take_features
from .infer import DEFAULT_CKPT, STANDARD_GAIT_HZ, load_model, run
from .poses import pose_fall, pose_ground, pose_walk

G = {g: i for i, g in enumerate(cfg.GROUPS)}
HOLDOUT_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "skeleton_tcn_holdout.pt")


class LiveClassifier(object):
    # confirm_s=0 : mesuré sur les montages réels (18 scénarios), la confirmation de 0.4 s
    # n'évitait aucun clignotement et coûtait 0.3 s de délai et 4 points d'accord.
    def __init__(self, model=None, buffer_s=6.0, ema=0.5, confirm_s=0.0, enter_p=0.55):
        self.model = model or load_model(DEFAULT_CKPT)
        self.buffer_s = buffer_s
        self.ema = ema
        self.confirm_s = confirm_s
        self.enter_p = enter_p
        self.buf = {n: collections.deque() for n in cfg.RX_NAMES}
        self.probs = None
        self.state, self.state_since = None, None
        self.cand, self.cand_since = None, None
        self.fall_t = None
        self.t_first = None

    def push(self, rx, t_ms, raw128):
        q = self.buf[rx]
        q.append((int(t_ms), np.asarray(raw128, np.int16)))
        while q and q[-1][0] - q[0][0] > self.buffer_s * 1000:
            q.popleft()

    def step(self, now_ms):
        """Classe l'instant now_ms. -> dict (ou None si pas encore de trames)."""
        rx = {}
        for n, q in self.buf.items():
            while q and q[0][0] < now_ms - self.buffer_s * 1000:
                q.popleft()
            if len(q) >= 5:
                rx[n] = (np.array([a for a, _ in q], np.int64), raw_to_complex(np.stack([b for _, b in q])))
        if not rx:
            return None
        if self.t_first is None:
            self.t_first = now_ms
        X, _, _ = take_features(rx, t_start_ms=now_ms - int(self.buffer_s * 1000), duration=self.buffer_s)
        out = run(self.model, X)
        p = np.exp(out["group_logp"][-1])
        self.probs = p if self.probs is None else self.ema * p + (1 - self.ema) * self.probs
        t = (now_ms - self.t_first) / 1000.0
        best = int(self.probs.argmax())
        event = None
        fallish = self.probs[G["chute"]] + self.probs[G["au_sol"]]
        warm = t < (self.model.receptive_field - 1) / cfg.FS
        if warm:                                   # pas assez de passé : aucune décision, aucune alerte
            return {"t": round(t, 2), "etat": "analyse", "brut": cfg.GROUPS[int(p.argmax())],
                    "probs": {g: round(float(v), 3) for g, v in zip(cfg.GROUPS, self.probs)},
                    "kp": np.zeros((17, 3)).tolist(), "demarrage": True,
                    "trames": {n: len(q) for n, q in self.buf.items()}, "evenement": None}
        if self.state is None:
            self.state, self.state_since = best, t
        elif fallish > 0.6 and self.state not in (G["chute"], G["au_sol"]):
            prev = self.state                                       # chute : alerte immédiate
            self.state, self.state_since, self.fall_t = G["chute"], t, t
            event = {"type": "alerte_chute", "t": round(t, 2), "de": cfg.GROUPS[prev]}
        elif best != self.state and self.probs[best] >= self.enter_p:
            if self.cand != best:
                self.cand, self.cand_since = best, t
            elif t - self.cand_since >= self.confirm_s:
                event = {"type": "transition", "t": round(t, 2), "de": cfg.GROUPS[self.state],
                         "vers": cfg.GROUPS[best]}
                self.state, self.state_since = best, t
                self.cand = None
        else:
            self.cand = None
        st = cfg.GROUPS[self.state]
        H = float(out["height"][-1])
        if st == "marche":
            kp = pose_walk((2 * np.pi * STANDARD_GAIT_HZ * t) % (2 * np.pi), H)
        elif st == "chute":
            al = min(1.0, (t - self.state_since) / 0.8)
            kp = pose_fall(al, H) if al < 1 else pose_ground(H)
        elif st == "vide":
            kp = np.zeros((17, 3))
        else:
            kp = out["kp"][-1]
        return {"t": round(t, 2), "etat": st, "brut": cfg.GROUPS[int(p.argmax())],
                "probs": {g: round(float(v), 3) for g, v in zip(cfg.GROUPS, self.probs)},
                "kp": np.round(np.asarray(kp, float), 3).tolist(), "demarrage": False,
                "trames": {n: len(q) for n, q in self.buf.items()}, "evenement": event}


# --------------------------------------------------------------------------
# Rejeu d'un scénario au rythme réel
# --------------------------------------------------------------------------
def replay(seq, clf, speed=1.0, hop_s=0.1, stop=None, realtime=True):
    """Générateur : pousse les trames du scénario au fil du temps et classe toutes les hop_s.
    Les trames sont dans le temps ESP ; on les rejoue à 'speed' fois la vitesse réelle."""
    frames = []
    for n, (ts, raw) in seq["rx"].items():
        frames += [(int(t), n, r) for t, r in zip(ts, raw)]
    frames.sort(key=lambda x: x[0])
    t0 = seq["t0_ms"]
    end = t0 + int(seq["duration"] * 1000)
    k = 0
    wall0 = time.time()
    now = t0
    while now <= end:
        if stop is not None and stop.is_set():
            return
        now += int(hop_s * 1000)
        while k < len(frames) and frames[k][0] <= now:
            clf.push(frames[k][1], frames[k][0], frames[k][2])
            k += 1
        r = clf.step(now)
        if r is not None:
            r["t"] = round((now - t0) / 1000.0, 2)
            yield r
        if realtime:
            delay = wall0 + (now - t0) / 1000.0 / speed - time.time()
            if delay > 0:
                time.sleep(delay)


def score(results, truth, skip_s=3.0):
    """Accord prédiction / vérité (hors démarrage), délais de détection des changements d'état."""
    def true_at(t):
        for a, b, g in truth:
            if a <= t < b:
                return g
        return truth[-1][2]
    ok = n = 0
    per = collections.defaultdict(lambda: [0, 0])
    for r in results:
        if r["t"] < skip_s:
            continue
        g = true_at(r["t"])
        good = r["etat"] == g or (g == "au_sol" and r["etat"] == "chute") or (g == "chute" and r["etat"] == "au_sol")
        ok += good
        n += 1
        per[g][0] += good
        per[g][1] += 1
    delays = []
    for i, (a, b, g) in enumerate(truth):
        if i == 0 or a < skip_s:
            continue
        hit = next((r["t"] for r in results if r["t"] >= a - 0.2 and
                    (r["etat"] == g or ({r["etat"], g} <= {"chute", "au_sol"}))), None)
        delays.append({"etat": g, "debut_verite_s": a,
                       "detecte_s": hit, "delai_s": None if hit is None else round(hit - a, 2)})
    fall_true = next((a for a, b, g in truth if g in ("chute", "au_sol")), None)
    fall_pred = next((r["t"] for r in results if r["evenement"] and r["evenement"]["type"] == "alerte_chute"), None)
    return {"accord": round(ok / max(n, 1), 3),
            "par_etat": {g: round(v[0] / max(v[1], 1), 3) for g, v in per.items()},
            "changements": delays,
            "chute_verite_s": fall_true, "chute_alerte_s": fall_pred,
            "fausse_alerte": bool(fall_pred is not None and (fall_true is None or fall_pred < fall_true - 0.5))}


# --------------------------------------------------------------------------
# Écoute UDP des vrais ESP32
# --------------------------------------------------------------------------
def udp_live(port, names, clf, hop_s=0.1):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "acquisition", "python"))
    from export_csv import lltf_standard_order          # noqa: E402
    from wisee_csi.parser import parse_line             # noqa: E402
    import queue
    q = queue.Queue()
    stop = threading.Event()
    from record_multi import udp_reader                 # noqa: E402
    threading.Thread(target=udp_reader, args=(port, names, q, stop), daemon=True).start()
    offset, prev_us, wraps = {}, {}, collections.Counter()
    last = time.time()
    try:
        while True:
            try:
                name, host_ts, line = q.get(timeout=0.05)
                fr = parse_line(line, host_ts)
                if fr is not None and fr.raw.size >= 128 and isinstance(fr.meta.get("local_timestamp"), int):
                    us = fr.meta["local_timestamp"]
                    if name in prev_us and us < prev_us[name] - 2 ** 31:
                        wraps[name] += 1                     # compteur 32 bits (µs) repassé par zéro
                    prev_us[name] = us
                    esp_ms = (us + wraps[name] * 2 ** 32) / 1000.0
                    offset.setdefault(name, host_ts * 1000 - esp_ms)          # horloge ESP calée sur le PC
                    sec = fr.meta.get("secondary_channel", 0)
                    clf.push(name, int(esp_ms + offset[name]), lltf_standard_order(fr.raw, sec))
            except queue.Empty:
                pass
            if time.time() - last >= hop_s:
                last = time.time()
                r = clf.step(int(time.time() * 1000))
                if r:
                    ev = r["evenement"]
                    print("\r%6.1fs  %-7s  %s   %s" % (r["t"], r["etat"].upper(),
                          " ".join("%s:%2d%%" % (g[:4], 100 * v) for g, v in r["probs"].items()),
                          ("<< %s" % ev) if ev else ""), end="", flush=True)
                    if ev:
                        print()
    except KeyboardInterrupt:
        stop.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="marche:4,debout:4,marche:4,chute:6")
    ap.add_argument("--source", choices=["simule", "reel"], default="simule")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--udp", type=int, help="écouter les vrais ESP32 sur ce port")
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    if a.udp:
        clf = LiveClassifier(load_model(a.ckpt or DEFAULT_CKPT))
        names = dict(m.split("=") for m in a.map)
        print("Écoute UDP :%d ... (Ctrl+C pour arrêter)" % a.udp)
        return udp_live(a.udp, names, clf)
    from .sequences import build
    clf = LiveClassifier(load_model(a.ckpt or (HOLDOUT_CKPT if os.path.exists(HOLDOUT_CKPT) else DEFAULT_CKPT)))
    seq = build(a.scenario, a.source, a.seed)
    print("Vérité :", " -> ".join("%s(%.1f-%.1fs)" % (g, s, e) for s, e, g in seq["truth"]))
    res = []
    for r in replay(seq, clf, a.speed):
        res.append(r)
        ev = r["evenement"]
        print("%5.1fs  %-7s %s" % (r["t"], r["etat"], ("<< %s" % ev) if ev else ""))
    print(score(res, seq["truth"]))


if __name__ == "__main__":
    main()
