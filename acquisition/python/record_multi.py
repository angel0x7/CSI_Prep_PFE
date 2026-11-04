# -*- coding: utf-8 -*-
"""
WISEE - Enregistreur CSI MULTI-RECEPTEURS (1 emetteur, N recepteurs).

Lit en parallele plusieurs flux CSI_DATA :
  --serial COM5:RX1 COM6:RX2 ...      un port serie par recepteur (csi_recv, 2 Mbauds)
  --udp 5005 [--map 192.168.4.201=RX1 ...]   tous les recepteurs envoient sur le meme port UDP,
                                      on les separe par leur adresse IP ; le PC envoie
                                      WISEE_HELLO a chaque IP de --map pour se faire connaitre

et pour chaque recepteur :
  - horodate chaque trame cote PC et l'ecrit dans data/multi_AAAAMMJJ_HHMMSS/<RX>.txt
    (meme format que record_csi.py : horodatage TAB ligne -> analyze_session.py marche dessus)
  - affiche chaque seconde : trames/s et pertes de chaque recepteur, sur une seule ligne
A la fin : un resume par recepteur + un controle de SYNCHRONISATION :
  l'emetteur numerote ses trames (champ 'id') et tous les recepteurs captent les memes trames,
  donc on mesure la part d'id communs a tous les recepteurs et le decalage d'arrivee sur le PC.

Exemples :
  python record_multi.py --serial COM5:RX1 COM6:RX2 --baud 2000000 --duration 60 --note "test 2 RX"
  python record_multi.py --udp 5005 --map 192.168.4.201=RX1 192.168.4.202=RX2 192.168.4.203=RX3 192.168.4.204=RX4 --duration 60
"""

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wisee_csi.parser import parse_line  # noqa: E402
from wisee_csi.metrics import StreamStats  # noqa: E402


def serial_reader(name, port, baud, q, stop):
    """Un thread par port serie : met (nom, horodatage, ligne) dans la file commune."""
    import serial  # pyserial
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.2)
    except Exception as e:  # port absent, deja ouvert par un autre programme...
        q.put((name, None, "ERREUR ouverture %s : %s" % (port, e)))
        return
    buf = b""
    with ser:
        while not stop.is_set():
            # lecture par blocs (bien plus rapide que readline, qui lit octet par octet)
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            now = time.time()
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                q.put((name, now, raw.decode("ascii", errors="replace")))


def udp_reader(port, names, q, stop):
    """Un seul thread UDP : separe les recepteurs par adresse IP (un tampon de ligne par IP)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.2)
    bufs = {}
    last_hello = 0.0
    while not stop.is_set():
        # Annonce du PC : chaque recepteur (csi_recv_udp) envoie ses trames a l'adresse
        # d'ou vient le dernier WISEE_HELLO ; on le renvoie chaque seconde.
        if time.time() - last_hello >= 1.0:
            for rx_ip in names:
                try:
                    sock.sendto(b"WISEE_HELLO", (rx_ip, port))
                except OSError:
                    pass
            last_hello = time.time()
        try:
            data, (ip, _) = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except ConnectionResetError:
            # Windows : un HELLO vers un recepteur eteint renvoie une erreur ICMP
            # qui ressort ici ; on l'ignore.
            continue
        now = time.time()
        name = names.get(ip, ip)                  # IP inconnue -> on la garde comme nom
        buf = bufs.get(name, "") + data.decode("ascii", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            q.put((name, now, line))
        bufs[name] = buf


def sync_report(ids_by_rx, ts_by_rx):
    """Part des trames (id emetteur) vues par TOUS les recepteurs + decalage d'arrivee sur le PC."""
    names = sorted(ids_by_rx)
    sets = {n: set(ids_by_rx[n]) for n in names}
    if not names or not all(sets.values()):
        return {}
    common = set.intersection(*sets.values())
    first, last = max(min(s) for s in sets.values()), min(max(s) for s in sets.values())
    expected = max(0, last - first + 1)          # trames emises pendant que tous ecoutaient
    rep = {"receivers": names,
           "ids_common_to_all": len(common),
           "ids_emitted_in_common_window": expected,
           "common_pct_of_emitted": round(100.0 * len(common) / expected, 2) if expected else None}
    ref = names[0]
    t_ref = dict(zip(ids_by_rx[ref], ts_by_rx[ref]))
    for n in names[1:]:
        t_n = dict(zip(ids_by_rx[n], ts_by_rx[n]))
        d = np.array([t_n[i] - t_ref[i] for i in common if i in t_n and i in t_ref]) * 1000
        if d.size:
            rep["pc_delay_ms_%s_vs_%s" % (n, ref)] = {"median": round(float(np.median(d)), 2),
                                                      "p95_abs": round(float(np.percentile(np.abs(d), 95)), 2)}
    return rep


def main():
    ap = argparse.ArgumentParser(description="Enregistreur CSI multi-recepteurs WISEE")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--serial", nargs="+", metavar="COMx:NOM", help="ex : COM5:RX1 COM6:RX2")
    src.add_argument("--udp", type=int, metavar="PORT")
    ap.add_argument("--map", nargs="*", default=[], metavar="IP=NOM", help="noms des recepteurs UDP")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--duration", type=float, default=0, help="secondes (0 = jusqu'a Ctrl+C)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--name", default="", help="nom du dossier de session (defaut : multi_AAAAMMJJ_HHMMSS)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-csv", action="store_true",
                    help="ne pas creer les RXn.csv (format equipe IA) a la fin de l'enregistrement")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sess_dir = os.path.join(args.outdir, args.name or "multi_%s" % stamp)
    if args.name and os.path.isdir(sess_dir) and os.listdir(sess_dir):
        sess_dir = sess_dir + "_" + stamp          # ne jamais ecraser une session existante
    os.makedirs(sess_dir, exist_ok=True)
    q, stop = queue.Queue(), threading.Event()
    threads, names = [], []

    if args.serial:
        for item in args.serial:
            port, _, name = item.partition(":")
            name = name or port
            names.append(name)
            threads.append(threading.Thread(target=serial_reader, daemon=True,
                                            args=(name, port, args.baud, q, stop)))
        source = "serial " + " ".join(args.serial) + " @%d" % args.baud
    else:
        ipmap = dict(m.split("=", 1) for m in args.map)
        names = list(ipmap.values())
        threads.append(threading.Thread(target=udp_reader, daemon=True, args=(args.udp, ipmap, q, stop)))
        source = "udp:%d %s" % (args.udp, " ".join(args.map))

    files, stats, ids, tss, rssi = {}, {}, {}, {}, {}

    def ensure(name):
        if name not in files:
            files[name] = open(os.path.join(sess_dir, "%s.txt" % name), "w")
            stats[name], ids[name], tss[name], rssi[name] = StreamStats(), [], [], []

    for n in names:
        ensure(n)
    for t in threads:
        t.start()

    print("Source : %s  ->  %s   (Ctrl+C pour arreter)" % (source, sess_dir))
    t0 = time.time()
    last_print = t0
    try:
        while True:
            now = time.time()
            if args.duration > 0 and now - t0 >= args.duration:
                break
            try:
                name, ts, line = q.get(timeout=0.2)
            except queue.Empty:
                name = None
            if name is not None:
                if ts is None:                       # erreur d'ouverture d'un port
                    print("\n" + line)
                    continue
                ensure(name)
                if line.strip():
                    fr = parse_line(line, host_ts=ts)
                    if fr is None:
                        if line.startswith("CSI_DATA"):
                            stats[name].bad_line()
                    else:
                        stats[name].update(fr)
                        files[name].write("%.6f\t%s\n" % (ts, line.strip()))
                        if isinstance(fr.seq, int):
                            ids[name].append(fr.seq)
                            tss[name].append(ts)
                        if isinstance(fr.meta.get("rssi"), int):
                            rssi[name].append(fr.meta["rssi"])
            if now - last_print >= 1.0:
                last_print = now
                parts = ["%s %3.0f/s %4.1f%%" % (n, stats[n].fps, stats[n].loss_pct) for n in sorted(stats)]
                sys.stdout.write("\r %6.1f s | " % (now - t0) + " | ".join(parts) + "   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    stop.set()
    for f in files.values():
        f.close()

    summary = {"source": source, "dir": sess_dir, "note": args.note, "date": stamp, "receivers": {}}
    print("\n\nResume par recepteur :")
    for n in sorted(stats):
        s = stats[n].summary()
        s["rssi_mean_dbm"] = round(float(np.mean(rssi[n])), 1) if rssi[n] else None
        summary["receivers"][n] = s
        print("  %-6s %5d trames | %5.1f trames/s | pertes %5.2f %% | mauvaises lignes %d | RSSI %s dBm"
              % (n, s["frames"], s["mean_fps"], s["loss_pct"], s["bad_lines"], s["rssi_mean_dbm"]))
    summary["sync"] = sync_report(ids, tss)
    if summary["sync"]:
        sy = summary["sync"]
        print("\nSynchronisation (compteur id de l'emetteur) :")
        print("  trames vues par TOUS les recepteurs : %d sur %d emises (%s %%)"
              % (sy["ids_common_to_all"], sy["ids_emitted_in_common_window"], sy["common_pct_of_emitted"]))
        for k, v in sy.items():
            if k.startswith("pc_delay"):
                print("  %s : mediane %s ms, |p95| %s ms" % (k, v["median"], v["p95_abs"]))
    with open(os.path.join(sess_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not args.no_csv:
        # export au format de l'equipe IA (timestamp_ms + 64 sous-porteuses LLTF), voir export_csv.py
        from export_csv import export_session
        print("Export CSV (format equipe IA) :")
        export_session(sess_dir, force=True)


if __name__ == "__main__":
    main()
