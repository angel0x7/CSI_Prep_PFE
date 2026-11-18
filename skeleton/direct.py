# -*- coding: utf-8 -*-
"""
Activité EN DIRECT dans la console, à partir des 4 récepteurs ESP32 (UDP).

On allume les cartes, on lance ce script, on fait des activités : la console affiche en continu
l'activité déduite (vide / debout / assis / marche / chute / au sol), les probabilités, l'état
des 4 récepteurs et l'historique des changements. Une chute déclenche une alerte.

    python -m skeleton.direct                       # vrais ESP32 (RX1..RX4 = 192.168.4.201..204, port 5005)
    python -m skeleton.direct --enregistrer         # + enregistre le CSI et les prédictions (réutilisable)
    python -m skeleton.direct --demo                # sans cartes : 4 faux ESP32 en local rejouent de vraies
                                                    #   prises jamais vues par le modèle de test
    python -m skeleton.direct --simple              # sortie ligne par ligne (sans couleurs ni rafraîchissement)

Touches pendant l'exécution (Windows) : d debout, a assis, m marche, c chute, v vide, x « je ne sais pas »,
q quitter. Elles notent ce que fait VRAIMENT la personne : avec --enregistrer, la session devient
une session étiquetée utilisable par finetune.py, et la console affiche l'accord modèle / réalité.

Moteur : celui de l'onglet « Test en direct » (live.LiveClassifier) : toutes les 100 ms, modèle causal
sur les 6 dernières secondes, lissage, machine à états. Aucune décision pendant les ~3 s de démarrage.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import queue
import socket
import sys
import threading
import time

import numpy as np

from . import config as cfg
from .infer import DEFAULT_CKPT, load_model
from .live import HOLDOUT_CKPT, LiveClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ACQ_PY = os.path.normpath(os.path.join(HERE, "..", "acquisition", "python"))
OUT_ROOT = os.path.normpath(os.path.join(HERE, "..", "acquisition", "data", "MAISON"))
DEFAULT_MAP = {"192.168.4.%d" % (200 + i): "RX%d" % i for i in range(1, 5)}       # firmware csi_recv_udp
DEMO_MAP = {"127.0.0.%d" % (10 + i): "RX%d" % i for i in range(1, 5)}
DEMO_SCENARIO = "debout:8,marche:8,assis:8,debout:6,marche:8,chute:8"
KEYS = {"d": "debout", "a": "assis", "m": "marche", "c": "chute", "v": "vide", "x": None}

# couleurs (palette de l'interface, variante sombre) ; texte en gris clair
RGB = {"vide": (111, 110, 104), "debout": (57, 135, 229), "assis": (25, 158, 112), "marche": (144, 133, 233),
       "chute": (230, 103, 103), "au_sol": (12, 163, 12), "analyse": (150, 150, 150)}


def ok_state(pred, truth):
    """La vérité 'chute' couvre la chute puis le temps au sol."""
    return pred == truth or (truth == "chute" and pred == "au_sol")


# --------------------------------------------------------------------------
# Réception UDP (même protocole que record_multi.py / session.py)
# --------------------------------------------------------------------------
class UdpCSI(object):
    def __init__(self, port, names):
        sys.path.insert(0, ACQ_PY)
        from export_csv import lltf_standard_order     # noqa: E402
        from record_multi import udp_reader            # noqa: E402
        from wisee_csi.parser import parse_line        # noqa: E402
        self._order, self._parse = lltf_standard_order, parse_line
        self.names = names
        self.q = queue.Queue()
        self.stop = threading.Event()
        threading.Thread(target=udp_reader, args=(port, names, self.q, self.stop), daemon=True).start()
        self.anchor, self.prev_us, self.wraps = {}, {}, collections.Counter()
        self.recent = {n: collections.deque() for n in cfg.RX_NAMES}     # heures PC des trames (1 s)
        self.last = {}
        self.total = collections.Counter()
        self.unknown = collections.Counter()                              # IP hors carte

    def poll(self, clf, writers=None):
        """Vide la file : décode, pousse dans le classifieur (et écrit les CSV si besoin)."""
        while True:
            try:
                name, host_ts, line = self.q.get_nowait()
            except queue.Empty:
                break
            i = line.find("CSI_DATA")
            if i < 0:
                continue
            fr = self._parse(line[i:], host_ts)
            if fr is None or fr.raw.size < 128 or not isinstance(fr.meta.get("local_timestamp"), int):
                continue
            if name not in self.recent:
                self.unknown[name] += 1
                continue
            us = fr.meta["local_timestamp"]
            if name in self.prev_us and us < self.prev_us[name] - 2 ** 31:
                self.wraps[name] += 1                                    # compteur 32 bits (µs) repassé par zéro
            self.prev_us[name] = us
            esp_ms = (us + self.wraps[name] * 2 ** 32) / 1000.0
            self.anchor.setdefault(name, host_ts * 1000.0 - esp_ms)      # horloge ESP calée sur le PC
            t_ms = int(round(esp_ms + self.anchor[name]))
            v = self._order(fr.raw, fr.meta.get("secondary_channel", 0))
            clf.push(name, t_ms, v)
            self.recent[name].append(host_ts)
            self.last[name] = host_ts
            self.total[name] += 1
            if writers is not None:
                writers[name].writerow([t_ms] + [int(x) for x in v])
        now = time.time()
        for d in self.recent.values():
            while d and d[0] < now - 1.0:
                d.popleft()

    def rates(self):
        return {n: len(d) for n, d in self.recent.items()}

    def close(self):
        self.stop.set()


# --------------------------------------------------------------------------
# Démo : 4 faux ESP32 en local qui rejouent un montage de vraies prises
# --------------------------------------------------------------------------
class DemoSender(object):
    def __init__(self, port, scenario, source, seed):
        from .sequences import build
        sys.path.insert(0, ACQ_PY)
        from wisee_csi.parser import format_line_s3     # noqa: E402
        self.fmt = format_line_s3
        self.seq = build(scenario, source, seed)
        self.port = port
        self.t_start = None
        self.done = threading.Event()
        frames = []
        for n, (ts, raw) in self.seq["rx"].items():
            for t, r in zip(ts, raw):
                frames.append((int(t) - self.seq["t0_ms"], n, r))
        frames.sort(key=lambda f: f[0])
        self.frames = frames
        self.socks = {}
        for ip, n in DEMO_MAP.items():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ip, 0))
            self.socks[n] = s

    def truth_at(self, t):
        for a, b, g in self.seq["truth"]:
            if a <= t < b:
                return g
        return None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        meta = {k: 0 for k in ("id", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
                               "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "ant",
                               "sig_len", "rx_state")}
        meta.update(mac="aa:bb:cc:dd:ee:ff", rssi=-50, bandwidth=1, channel=11, secondary_channel=1)
        self.t_start = time.time()
        for k, (t, n, r) in enumerate(self.frames):
            wait = self.t_start + t / 1000.0 - time.time()
            if wait > 0:
                time.sleep(wait)
            meta["id"] = k
            meta["local_timestamp"] = int(t * 1000) & 0xFFFFFFFF
            line = self.fmt(meta, np.asarray(r, np.int16)) + "\n"
            try:
                self.socks[n].sendto(line.encode("ascii"), ("127.0.0.1", self.port))
            except OSError:
                pass
        self.done.set()


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
class Screen(object):
    def __init__(self, simple):
        self.simple = simple
        if not simple:
            os.system("")                               # active les séquences ANSI de la console Windows
            sys.stdout.write("\x1b[2J\x1b[?25l")

    @staticmethod
    def col(text, g, bold=False):
        r, gg, b = RGB.get(g, (200, 200, 200))
        return "\x1b[%s38;2;%d;%d;%dm%s\x1b[0m" % ("1;" if bold else "", r, gg, b, text)

    def draw(self, lines):
        sys.stdout.write("\x1b[H" + "".join(l + "\x1b[K\n" for l in lines) + "\x1b[J")
        sys.stdout.flush()

    def close(self):
        if not self.simple:
            sys.stdout.write("\x1b[?25h\x1b[0m\n")
            sys.stdout.flush()


def fmt_dur(s):
    s = int(max(s, 0))
    return "%02d:%02d" % (s // 60, s % 60) if s < 3600 else "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)


def read_key():
    try:
        import msvcrt
    except ImportError:
        return None
    if msvcrt.kbhit():
        k = msvcrt.getwch()
        if k in ("\x00", "\xe0"):                       # touche spéciale : on lit et on ignore le 2e code
            msvcrt.getwch()
            return None
        return k.lower()
    return None


# --------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udp", type=int, default=5005, help="port UDP (défaut 5005)")
    ap.add_argument("--map", nargs="*", default=None, metavar="IP=RXn",
                    help="adresses des récepteurs (défaut : 192.168.4.201..204 = RX1..RX4)")
    ap.add_argument("--ckpt", default=None, help="modèle (défaut : skeleton_tcn.pt ; en démo : modèle de test)")
    ap.add_argument("--enregistrer", action="store_true", help="enregistre CSI + prédictions dans acquisition/data/MAISON")
    ap.add_argument("--lieu", default="maison")
    ap.add_argument("--personne", default="sujet1")
    ap.add_argument("--simple", action="store_true", help="sortie texte ligne par ligne")
    ap.add_argument("--demo", nargs="?", const="reel", choices=["reel", "simule"],
                    help="sans cartes : rejoue un montage (reel = vraies prises mises de côté)")
    ap.add_argument("--scenario", default=DEMO_SCENARIO, help="scénario de la démo, ex. 'marche:8,chute:8'")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    # ------------------------------------------------------------ modèle et sources
    demo = None
    if a.demo:
        names = dict(DEMO_MAP)
        ckpt = a.ckpt or (HOLDOUT_CKPT if a.demo == "reel" and os.path.exists(HOLDOUT_CKPT) else DEFAULT_CKPT)
        print("Préparation de la démo (montage %s : %s) ..." % (a.demo, a.scenario))
        demo = DemoSender(a.udp, a.scenario, a.demo, a.seed)
    else:
        names = dict(m.split("=") for m in a.map) if a.map else dict(DEFAULT_MAP)
        ckpt = a.ckpt or DEFAULT_CKPT
        if sorted(names.values()) != sorted(cfg.RX_NAMES):
            print("ATTENTION : --map devrait nommer RX1..RX4 (reçu : %s)" % sorted(names.values()))
    model = load_model(ckpt)
    clf = LiveClassifier(model)
    warm_s = (model.receptive_field - 1) / cfg.FS
    try:
        src = UdpCSI(a.udp, names)
    except OSError as e:
        raise SystemExit("Port UDP %d indisponible (%s). Un autre script écoute-t-il déjà ? (viewer, record_multi...)"
                         % (a.udp, e))

    # ------------------------------------------------------------ enregistrement
    out_dir, writers, files, pred_w = None, None, [], None
    t0 = time.time()
    if a.enregistrer:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUT_ROOT, "direct_%s%s" % (stamp, "_demo" if demo else ""))
        os.makedirs(out_dir, exist_ok=True)
        writers = {}
        for n in cfg.RX_NAMES:
            f = open(os.path.join(out_dir, n + ".csv"), "w", newline="")
            files.append(f)
            writers[n] = csv.writer(f)
        f = open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8")
        files.append(f)
        pred_w = csv.writer(f)
        pred_w.writerow(["t_s", "heure", "etat", "verite_declaree"] + cfg.GROUPS + ["trames_s_" + n for n in cfg.RX_NAMES])

    scr = Screen(a.simple)
    if demo:
        demo.start()
        t0 = demo.t_start or time.time()

    # ------------------------------------------------------------ boucle
    history = collections.deque(maxlen=8)
    declared, decl_since, segments = None, None, []                 # vérité tapée au clavier
    agree = collections.Counter()
    r = None
    last_step = last_draw = last_simple = 0.0
    alert = None
    cur_state, state_since = None, 0.0
    try:
        while True:
            now = time.time()
            src.poll(clf, writers)
            if demo and demo.t_start:
                t0 = demo.t_start
            t = now - t0

            k = read_key()
            if k == "q":
                break
            if k in KEYS:
                if declared is not None:
                    segments.append([round(decl_since, 2), round(t, 2), declared])
                declared, decl_since = KEYS[k], t
                history.appendleft((t, "vous : %s" % (declared.upper() if declared else "je ne sais pas"), None))

            if now - last_step >= 0.1:
                last_step = now
                r = clf.step(int(now * 1000))
                if r is not None:
                    if r["etat"] != cur_state:
                        cur_state, state_since = r["etat"], t
                    ev = r["evenement"]
                    if ev:
                        if ev["type"] == "alerte_chute":
                            alert = (t, ev["de"])
                            history.appendleft((t, "ALERTE CHUTE (depuis %s)" % ev["de"], "chute"))
                        else:
                            history.appendleft((t, "%s -> %s" % (ev["de"], ev["vers"]), ev["vers"]))
                        if a.simple:
                            print("[%s] %s" % (fmt_dur(t), history[0][1]))
                    if r["etat"] not in ("chute", "au_sol"):
                        alert = None
                    truth = demo.truth_at(t) if demo else declared
                    if truth and r["etat"] != "analyse":
                        agree["n"] += 1
                        agree["ok"] += ok_state(r["etat"], truth)
                    if pred_w is not None:
                        rt = src.rates()
                        pred_w.writerow([round(t, 2), datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                         r["etat"], declared or ""] + [r["probs"][g] for g in cfg.GROUPS]
                                        + [rt[n] for n in cfg.RX_NAMES])

            if a.simple:
                if now - last_simple >= 1.0:
                    last_simple = now
                    rt = src.rates()
                    st = r["etat"] if r else "en attente des trames"
                    print("[%s] %-8s %s | trames/s %s" % (
                        fmt_dur(t), st.upper(),
                        " ".join("%s %2d%%" % (g, 100 * v) for g, v in r["probs"].items()) if r else "",
                        " ".join("%s:%d" % (n, rt[n]) for n in cfg.RX_NAMES)), flush=True)
            elif now - last_draw >= 0.2:
                last_draw = now
                scr.draw(render(a, ckpt, src, r, t, warm_s, history, alert, declared, agree, demo, out_dir,
                                state_since, scr))
            if demo and demo.done.is_set() and time.time() - demo.t_start > demo.seq["duration"] + 1.5:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        scr.close()
    t_end = time.time() - t0
    if declared is not None:
        segments.append([round(decl_since, 2), round(t_end, 2), declared])

    # ------------------------------------------------------------ fin
    print("Arrêt après %s. Trames reçues : %s" % (fmt_dur(t_end), dict((n, src.total[n]) for n in cfg.RX_NAMES)))
    if agree["n"]:
        print("Accord modèle / %s : %d %% (%d instants, démarrage exclu)" % (
            "vérité de la démo" if demo else "activités déclarées au clavier", 100 * agree["ok"] / agree["n"], agree["n"]))
    if out_dir:
        for f in files:
            f.close()
        meta = {"t0_ms": int(t0 * 1000), "consignes": segments, "reaction_s": 1.5, "lieu": a.lieu,
                "personne": a.personne, "protocole": "libre (direct.py, étiquettes tapées au clavier)",
                "modele": os.path.basename(ckpt), "carte_udp": names,
                "trames": {n: src.total[n] for n in cfg.RX_NAMES}, "duree_s": round(t_end, 2),
                "remarque": "étiquettes = touches tapées pendant la session ; 'chute' = chute puis au sol"}
        with open(os.path.join(out_dir, "consignes.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print("Enregistré dans", out_dir)
        if not segments:
            print("(aucune activité tapée au clavier : session non étiquetée, inutilisable pour finetune.py)")


def render(a, ckpt, src, r, t, warm_s, history, alert, declared, agree, demo, out_dir, state_since, scr):
    C = scr.col
    L = []
    L.append("\x1b[1mWISEE · activité en direct (CSI WiFi, 4 récepteurs)\x1b[0m    %s   durée %s"
             % (datetime.datetime.now().strftime("%H:%M:%S"), fmt_dur(t)))
    L.append("\x1b[2mmodèle %s%s · port UDP %d%s\x1b[0m" % (
        os.path.basename(ckpt), " (n'a jamais vu ces prises)" if demo and "holdout" in ckpt else "", a.udp,
        " · DÉMO : faux ESP32 locaux" if demo else ""))
    L.append("")

    # récepteurs
    rt = src.rates()
    cells, missing = [], []
    for n in cfg.RX_NAMES:
        last = src.last.get(n)
        if rt[n] >= 20:
            cells.append(C("%s %3d/s ✓" % (n, rt[n]), "assis"))
        elif last is None:
            cells.append(C("%s   -   ✗" % n, "chute"))
            missing.append(n)
        else:
            cells.append(C("%s %3d/s ! (silence %.0f s)" % (n, rt[n], time.time() - last), "chute"))
            missing.append(n)
    L.append("Récepteurs   " + "    ".join(cells))
    if src.unknown:
        L.append(C("  trames d'adresses inconnues : %s (vérifier --map)" % ", ".join(src.unknown), "chute"))
    if missing and src.last:
        L.append(C("  ! %s absent(s) : le modèle a été entraîné avec les 4 récepteurs, résultats peu fiables."
                   % ", ".join(missing), "chute"))
    L.append("")

    # état
    if r is None:
        if not src.last:
            L.append("  En attente des trames UDP ...")
            if t > 5 and not demo:
                L.append("  Vérifier : PC connecté au Wi-Fi WISEE_CSI, émetteur S allumé, pare-feu UDP %d ouvert," % a.udp)
                L.append("  ping 192.168.4.201 répond, aucun autre script n'écoute le port.")
        else:
            L.append("  Réception en cours ...")
        L += [""] * 9
    else:
        st = r["etat"]
        if alert:
            L.append(C("  ███  CHUTE DÉTECTÉE  (à %s, depuis %s)  ███" % (fmt_dur(alert[0]), alert[1]), "chute", True))
        else:
            L.append("")
        if r["demarrage"]:
            left = warm_s - r["t"]
            L.append("  ÉTAT :  " + C("ANALYSE", "analyse", True) + "   démarrage, encore %.1f s de signal nécessaire" % max(left, 0))
        else:
            L.append("  ÉTAT :  " + C(st.upper().replace("_", " "), st, True)
                     + "   depuis %.0f s · confiance %d %%" % (t - state_since, 100 * r["probs"].get(st, 0)))
        truth = demo.truth_at(t) if demo else declared
        if truth or (not demo and declared is None):
            if demo:
                L.append("  vérité de la démo : %s" % C(truth.upper(), truth))
            elif declared:
                L.append("  vous avez déclaré : %s" % C(declared.upper(), declared))
            else:
                L.append("\x1b[2m  (touches d/a/m/c/v pour noter ce que fait la personne)\x1b[0m")
        else:
            L.append("")
        L.append("")
        for g in cfg.GROUPS:
            v = r["probs"][g]
            n = int(round(v * 30))
            L.append("  %-7s %s%s %3d %%" % (g, C("█" * n, g), "\x1b[2m" + "·" * (30 - n) + "\x1b[0m", 100 * v))
    L.append("")
    if agree["n"]:
        L.append("Accord avec la %s : %d %%" % ("vérité de la démo" if demo else "réalité déclarée", 100 * agree["ok"] / agree["n"]))
    L.append("\x1b[1mHistorique\x1b[0m")
    for (tt, txt, g) in list(history)[:6]:
        L.append("  %s  %s" % (fmt_dur(tt), C(txt, g) if g else txt))
    L.append("")
    L.append("\x1b[2mq ou Ctrl+C : arrêter%s\x1b[0m" % ((" · enregistrement : " + out_dir) if out_dir else ""))
    return L


if __name__ == "__main__":
    main()
