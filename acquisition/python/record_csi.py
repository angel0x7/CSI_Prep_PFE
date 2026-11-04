# -*- coding: utf-8 -*-
"""
WISEE - Enregistreur CSI.

Lit le flux CSI_DATA depuis :
  --udp PORT        le simulateur (ou plus tard notre firmware qui enverra en UDP)
  --serial COM5     un vrai ESP32 flashe avec esp-csi/csi_recv (921600 bauds)
  --file chemin     un fichier deja enregistre (rejeu)

et :
  - horodate chaque trame cote PC
  - affiche chaque seconde : trames/s, pertes, doublons, RSSI, nombre de sous-porteuses
  - sauvegarde la session brute (data/session_AAAAMMJJ_HHMMSS.txt : horodatage TAB ligne)
  - affiche un resume (a reprendre tel quel dans le plan de validation)

Exemples :
  python record_csi.py --udp 5005 --duration 60
  python record_csi.py --serial COM5 --duration 300 --note "piece vide, salle B204"
"""

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wisee_csi.parser import parse_line  # noqa: E402
from wisee_csi.metrics import StreamStats  # noqa: E402


def udp_lines(port, stop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    buf = ""
    while not stop():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            yield None
            continue
        buf += data.decode("ascii", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line


def serial_lines(port, baud, stop):
    try:
        import serial  # pyserial
    except ImportError:
        raise SystemExit("pyserial manquant : pip install pyserial")
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.5)
    while not stop():
        raw = ser.readline()
        if not raw:
            yield None
            continue
        yield raw.decode("ascii", errors="replace")


def file_lines(path, stop):
    with open(path) as f:
        for line in f:
            if stop():
                break
            # nos fichiers de session : "horodatage<TAB>ligne" -> on garde l'horodatage d'origine
            if "\t" in line:
                ts, line = line.split("\t", 1)
                yield (float(ts), line)
            else:
                yield line


def main():
    ap = argparse.ArgumentParser(description="Enregistreur CSI WISEE")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--udp", type=int, metavar="PORT")
    src.add_argument("--serial", metavar="COMx")
    src.add_argument("--file", metavar="CHEMIN")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--duration", type=float, default=0, help="secondes (0 = jusqu'a Ctrl+C)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--note", default="", help="description de la session (piece, personnes...)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.outdir, "session_%s.txt" % stamp)
    t_start = time.time()
    stop = (lambda: args.duration > 0 and time.time() - t_start >= args.duration)

    if args.udp:
        lines, source = udp_lines(args.udp, stop), "udp:%d" % args.udp
    elif args.serial:
        lines, source = serial_lines(args.serial, args.baud, stop), "serial:%s@%d" % (args.serial, args.baud)
    else:
        lines, source = file_lines(args.file, stop), "file:%s" % args.file

    stats = StreamStats()
    n_sub = None
    rssi_sum = rssi_n = 0
    last_print = time.time()
    print("Source : %s  ->  %s   (Ctrl+C pour arreter)" % (source, out_path))
    with open(out_path, "w") as out:
        try:
            for line in lines:
                now = time.time()
                if isinstance(line, tuple):            # rejeu d'une session : horodatage d'origine
                    now, line = line
                if line is not None and line.strip():
                    frame = parse_line(line, host_ts=now)
                    if frame is None:
                        if line.startswith("CSI_DATA"):
                            stats.bad_line()           # ligne CSI tronquee / corrompue
                    else:
                        stats.update(frame)
                        out.write("%.6f\t%s\n" % (now, line.strip()))
                        n_sub = frame.csi.size
                        if isinstance(frame.meta.get("rssi"), int):
                            rssi_sum += frame.meta["rssi"]
                            rssi_n += 1
                if not args.file and now - last_print >= 1.0:
                    last_print = now
                    sys.stdout.write("\r %6.1f s | %5.0f trames/s | total %7d | pertes %5.2f %% | "
                                     "doublons %d | mauvaises lignes %d | RSSI %5.1f dBm | %s sous-porteuses   "
                                     % (now - t_start, stats.fps, stats.n_frames, stats.loss_pct,
                                        stats.n_dup, stats.n_bad_lines,
                                        rssi_sum / rssi_n if rssi_n else float("nan"), n_sub))
                    sys.stdout.flush()
        except KeyboardInterrupt:
            pass

    summary = stats.summary()
    summary.update({"source": source, "file": out_path, "note": args.note,
                    "rssi_mean_dbm": round(rssi_sum / rssi_n, 1) if rssi_n else None,
                    "subcarriers": n_sub, "date": stamp})
    with open(out_path.replace(".txt", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n\nResume de la session :")
    for k, v in summary.items():
        print("  %-18s %s" % (k, v))


if __name__ == "__main__":
    main()
