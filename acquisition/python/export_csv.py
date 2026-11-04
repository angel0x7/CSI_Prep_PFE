# -*- coding: utf-8 -*-
"""
WISEE - export CSV pour l'equipe IA (format du depot CSI_Prep_PFE).

Chaque fichier brut <RX>.txt d'une session (ecrit par record_multi.py ou record_csi.py)
donne un fichier <RX>.csv au format du dataset ESP32 utilise par l'equipe IA :

    timestamp_ms, v0, v1, ..., v127          (129 colonnes, pas d'en-tete)

  - timestamp_ms : horloge de l'ESP32 (local_timestamp, en microsecondes) convertie en ms,
                   recalee sur l'heure du PC de la premiere trame -> heure Unix en ms,
                   reguliere (pas de gigue reseau) ; le retour a zero du compteur
                   (toutes les 71,6 min) est corrige.
  - v0..v127     : les 64 sous-porteuses du LLTF, pour chacune la partie IMAGINAIRE puis
                   la partie REELLE, dans l'ordre ESP32 classique 0..31, -32..-1
                   (meme ordre que le dataset de l'equipe IA).
                   NB : en HT40 (secondaire en dessous), l'ESP32-S3 rend le LLTF dans l'ordre
                   -32..31 (DC au milieu). On remet donc les deux moities dans l'ordre standard.

Le fichier .txt reste la reference : il contient en plus la partie HT-LTF (114 sous-porteuses
utiles), le RSSI et le numero de trame de l'emetteur (synchronisation entre recepteurs).

Usage :
  python export_csv.py data/2026-11-05_4RX_UDP/marche_S-RX3      (une session)
  python export_csv.py data                                     (toutes les sessions, recursif)
  python export_csv.py data --force                             (refait les CSV deja existants)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wisee_csi.parser import parse_line  # noqa: E402

N_LLTF_VALUES = 128          # 64 sous-porteuses x (imaginaire, reel)
WRAP_US = 2 ** 32            # local_timestamp est un compteur 32 bits en microsecondes


def lltf_standard_order(raw, secondary_channel):
    """Renvoie les 128 valeurs du LLTF dans l'ordre 0..31, -32..-1 (paires im/re conservees)."""
    v = raw[:N_LLTF_VALUES]
    if secondary_channel == 2:                     # HT40, secondaire en dessous : ordre -32..31
        v = np.concatenate([v[64:], v[:64]])       # on echange les moities (32 paires chacune)
    return v


def export_file(txt_path, csv_path):
    """Convertit un fichier brut en CSV. Renvoie le nombre de trames ecrites."""
    rows = []
    t0_host = None
    t0_esp = None
    prev_esp = None
    wraps = 0
    with open(txt_path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            host_ts = None
            if "\t" in line:                       # format "horodatage_PC <TAB> ligne CSI_DATA"
                ts_txt, line = line.split("\t", 1)
                try:
                    host_ts = float(ts_txt)
                except ValueError:
                    host_ts = None
            fr = parse_line(line, host_ts=host_ts)
            if fr is None or fr.raw.size < N_LLTF_VALUES:
                continue
            esp = fr.meta.get("local_timestamp")
            if not isinstance(esp, int):
                continue
            if prev_esp is not None and esp < prev_esp - WRAP_US // 2:
                wraps += 1                         # le compteur 32 bits est repasse par zero
            prev_esp = esp
            esp_unwrapped = esp + wraps * WRAP_US
            if t0_esp is None:
                t0_esp = esp_unwrapped
                t0_host = host_ts if host_ts is not None else 0.0
            t_ms = int(round(t0_host * 1000.0 + (esp_unwrapped - t0_esp) / 1000.0))
            sec = fr.meta.get("secondary_channel", 0)
            rows.append([t_ms] + [int(x) for x in lltf_standard_order(fr.raw, sec)])

    with open(csv_path, "w", encoding="ascii", newline="") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return len(rows)


def find_raw_files(root):
    """Fichiers bruts a convertir : RX*.txt d'une session, ou session_*.txt (1 recepteur)."""
    if os.path.isfile(root):
        return [root]
    out = []
    for d, _, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(".txt") and (name.startswith("RX") or name.startswith("session_")):
                out.append(os.path.join(d, name))
    return out


def export_session(path, force=False, quiet=False):
    """Exporte tous les fichiers bruts sous 'path'. Renvoie la liste (csv, nb_trames)."""
    done = []
    for txt in find_raw_files(path):
        csv_path = os.path.splitext(txt)[0] + ".csv"
        if os.path.exists(csv_path) and not force:
            continue
        n = export_file(txt, csv_path)
        done.append((csv_path, n))
        if not quiet:
            print("  %-70s %6d trames" % (os.path.relpath(csv_path), n))
    return done


def main():
    ap = argparse.ArgumentParser(description="Export CSV (format equipe IA) des sessions WISEE")
    ap.add_argument("chemin", help="dossier de session, dossier data/ entier, ou fichier .txt")
    ap.add_argument("--force", action="store_true", help="refaire les CSV deja presents")
    args = ap.parse_args()
    done = export_session(args.chemin, force=args.force)
    print("%d fichier(s) CSV ecrit(s)." % len(done))


if __name__ == "__main__":
    main()
