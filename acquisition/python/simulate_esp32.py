# -*- coding: utf-8 -*-
"""
WISEE - Simulateur "faux ESP32-S3".

Envoie en UDP des lignes CSI_DATA au format EXACT du firmware espressif/esp-csi
(csi_recv, cible esp32s3, HT40 : LLTF 64 + HT-LTF 128 sous-porteuses = len 384),
a 100 trames/s par defaut, en suivant un scenario d'activites. Il ecrit aussi la
verite terrain (debut, fin, activite) dans un fichier labels CSV.

Exemples :
  python simulate_esp32.py                                   # scenario par defaut, temps reel
  python simulate_esp32.py --scenario "empty:10,walk:10,sit:15,fall:2,lying:15"
  python simulate_esp32.py --loss 0.02 --jitter-ms 3         # pertes et irregularites
  python simulate_esp32.py --fast --out data/sim.txt         # genere un fichier sans attendre
"""

import argparse
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wisee_csi.channel_model import Room, HumanTrajectory, csi_buffer, STATES  # noqa: E402
from wisee_csi.parser import format_line_s3  # noqa: E402

DEFAULT_SCENARIO = "empty:10,walk:10,sit:15,stand:5,fall:2,lying:15"


def parse_scenario(text):
    segs = []
    for item in text.split(","):
        name, dur = item.strip().split(":")
        if name not in STATES:
            raise SystemExit("Etat inconnu '%s'. Etats possibles : %s" % (name, ", ".join(STATES)))
        segs.append((name, float(dur)))
    return segs


def main():
    ap = argparse.ArgumentParser(description="Faux ESP32-S3 : flux CSI_DATA en UDP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--rate", type=float, default=100.0, help="trames/s (esp-csi : 100)")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO,
                    help="liste etat:duree_s ; etats : " + ", ".join(STATES))
    ap.add_argument("--loss", type=float, default=0.0, help="probabilite de perte par trame")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="ecart-type du retard d'envoi")
    ap.add_argument("--seed", type=int, default=0, help="graine (meme graine = meme piece)")
    ap.add_argument("--labels", default="data/sim_labels.csv", help="fichier verite terrain")
    ap.add_argument("--out", default=None, help="ecrit aussi les lignes dans ce fichier")
    ap.add_argument("--fast", action="store_true", help="pas d'attente : generation hors ligne")
    ap.add_argument("--no-udp", action="store_true", help="n'envoie rien sur le reseau")
    args = ap.parse_args()

    segments = parse_scenario(args.scenario)
    total = sum(d for _, d in segments)
    rng = np.random.default_rng(args.seed + 100)
    room = Room(seed=args.seed)
    human = HumanTrajectory(seed=args.seed + 1)
    dt = 1.0 / args.rate

    for path in (args.labels, args.out):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    out = open(args.out, "w") if args.out else None
    sock = None if args.no_udp else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with open(args.labels, "w") as lab:
        lab.write("start_s,end_s,label\n")
        t0 = 0.0
        for name, dur in segments:
            lab.write("%.3f,%.3f,%s\n" % (t0, t0 + dur, name))
            t0 += dur

    print("Scenario (%.0f s) : %s" % (total, args.scenario))
    if sock:
        print("Envoi UDP vers %s:%d a %.0f trames/s  (Ctrl+C pour arreter)" % (args.host, args.port, args.rate))

    meta = {"id": 0, "mac": "1a:00:00:00:00:00", "rssi": -40, "rate": 11, "sig_mode": 1,
            "mcs": 0, "bandwidth": 1, "smoothing": 0, "not_sounding": 1, "aggregation": 0,
            "stbc": 0, "fec_coding": 0, "sgi": 0, "noise_floor": -93, "ampdu_cnt": 0,
            "channel": 11, "secondary_channel": 2, "local_timestamp": 0, "ant": 0,
            "sig_len": 47, "rx_state": 1, "first_word": 0}

    start = time.time()
    n_sent = n_dropped = 0
    k = 0
    seg_i, seg_t0 = 0, 0.0
    try:
        while True:
            t = k * dt
            if t >= total:
                break
            while t >= seg_t0 + segments[seg_i][1]:
                seg_t0 += segments[seg_i][1]
                seg_i += 1
            state = segments[seg_i][0]
            h_amp, h_dist = human.sample(state, t, t - seg_t0, dt)
            raw = csi_buffer(room, h_amp, h_dist, rng)
            meta["id"] = k                                   # compteur de l'emetteur
            meta["local_timestamp"] = int(t * 1e6)
            meta["rssi"] = int(round(-42 - 3 * h_amp + rng.normal(0, 1)))
            line = format_line_s3(meta, raw)
            k += 1

            if rng.random() < args.loss:
                n_dropped += 1
                continue
            if not args.fast:
                target = start + t + (rng.normal(0, args.jitter_ms / 1000.0) if args.jitter_ms else 0.0)
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)
            if sock:
                sock.sendto((line + "\n").encode("ascii"), (args.host, args.port))
            if out:
                out.write(line + "\n")
            n_sent += 1
            if not args.fast and k % int(args.rate) == 0:
                sys.stdout.write("\r t=%5.1f s  etat=%-6s  envoyees=%d  perdues=%d   "
                                 % (t, state, n_sent, n_dropped))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()
    print("\nTermine : %d trames envoyees, %d perdues volontairement. Labels : %s"
          % (n_sent, n_dropped, args.labels))


if __name__ == "__main__":
    main()
