# -*- coding: utf-8 -*-
"""
Session GUIDÉE avec les vrais ESP32 : test du modèle en direct + enregistrement de données étiquetées.

L'ordinateur annonce les consignes (texte + bip) ; la personne les suit ; pendant ce temps le modèle
classe l'état en direct, et TOUT est enregistré (CSI des 4 récepteurs au format du DATASET + consignes
= étiquettes + prédictions). À la fin : bilan (accord, chutes détectées, fausses alertes).
Les sessions servent ensuite à affiner le modèle sur le nouvel environnement (finetune.py).

    python -m skeleton.session --udp 5005 --map 192.168.4.201=RX1 192.168.4.202=RX2 192.168.4.203=RX3 192.168.4.204=RX4
    python -m skeleton.session ... --protocole court          (environ 2 min, pour un premier essai)
    python -m skeleton.session ... --scenario "vide:20,debout:15,marche:15,assis:15,chute:15"

Sorties : acquisition/data/MAISON/session_AAAAMMJJ_HHMMSS/
    RX1.csv .. RX4.csv   même format que le DATASET (timestamp_ms + 128 valeurs LLTF)
    consignes.json       segments [début_s, fin_s, état] (= étiquettes), t0_ms, lieu, personne...
    predictions.csv      t, état détecté, probabilités (modèle en direct)
    bilan.json           résultat du test

SÉCURITÉ : pour 'chute', se laisser tomber doucement sur un matelas ou un tapis épais.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import queue
import sys
import threading
import time

import numpy as np

from . import config as cfg
from .infer import DEFAULT_CKPT, load_model
from .live import LiveClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.normpath(os.path.join(HERE, "..", "acquisition", "data", "MAISON"))

PROTOCOLES = {
    # ~2 min : premier essai rapide
    "court": "vide:15,debout:12,marche:12,assis:12,debout:8,marche:12,chute:15",
    # ~4 min 30 : session standard (à répéter 3-5 fois, en variant l'endroit dans la pièce)
    "standard": ("vide:20,debout:15,marche:15,assis:20,debout:10,marche:15,debout:10,assis:15,"
                 "marche:15,debout:10,marche:15,assis:15,debout:10,marche:12,chute:20"),
    # chutes répétées (sur matelas) : vraies transitions marche/debout -> chute
    "chutes": "debout:10,marche:10,chute:15",
}
CONSIGNES = {
    "vide": "SORTEZ DE LA ZONE (pièce vide)",
    "debout": "RESTEZ DEBOUT, immobile",
    "assis": "ASSEYEZ-VOUS (chaise), immobile",
    "marche": "MARCHEZ (allers-retours)",
    "chute": "TOMBEZ (sur le matelas) puis RESTEZ AU SOL",
}
REACTION_S = 1.5            # temps laissé pour suivre une consigne (ignoré dans le bilan / l'entraînement)


def parse(text):
    out = []
    for part in text.replace(" ", "").split(","):
        s, _, d = part.partition(":")
        if s not in CONSIGNES:
            raise SystemExit("état inconnu : %r (choix : %s)" % (s, ", ".join(CONSIGNES)))
        out.append((s, float(d)))
    if any(s == "chute" for s, _ in out[:-1]):
        raise SystemExit("'chute' doit être le dernier état d'une session")
    return out


def beep(kind="go"):
    try:
        import winsound
        winsound.Beep(1200 if kind == "go" else 700, 180 if kind == "go" else 90)
    except Exception:
        print("\a", end="", flush=True)


def expected_ok(pred, state):
    if state == "chute":
        return pred in ("chute", "au_sol")
    return pred == state


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udp", type=int, default=5005)
    ap.add_argument("--map", nargs="*", default=[], metavar="IP=RXn")
    ap.add_argument("--protocole", choices=sorted(PROTOCOLES), default="standard")
    ap.add_argument("--scenario", help="remplace le protocole, ex. 'debout:15,marche:15,chute:15'")
    ap.add_argument("--lieu", default="maison", help="nom de l'environnement (noté dans consignes.json)")
    ap.add_argument("--personne", default="sujet1")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--attente", type=float, default=10.0, help="secondes avant la 1re consigne (démarrage du modèle)")
    a = ap.parse_args()

    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "acquisition", "python")))
    from export_csv import lltf_standard_order          # noqa: E402
    from record_multi import udp_reader                 # noqa: E402
    from wisee_csi.parser import parse_line             # noqa: E402

    script = parse(a.scenario or PROTOCOLES[a.protocole])
    total = sum(d for _, d in script)
    names = dict(m.split("=") for m in a.map)
    if names and sorted(names.values()) != sorted(cfg.RX_NAMES):
        print("ATTENTION : --map devrait nommer RX1..RX4 (reçu : %s)" % sorted(names.values()))
    clf = LiveClassifier(load_model(a.ckpt))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(a.out, "session_%s" % stamp)
    os.makedirs(out_dir, exist_ok=True)

    q = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=udp_reader, args=(a.udp, names, q, stop), daemon=True).start()
    rows = collections.defaultdict(list)
    anchor, prev_us, wraps = {}, {}, collections.Counter()

    def drain():
        while True:
            try:
                name, host_ts, line = q.get_nowait()
            except queue.Empty:
                return
            fr = parse_line(line, host_ts)
            if fr is None or fr.raw.size < 128 or not isinstance(fr.meta.get("local_timestamp"), int):
                continue
            us = fr.meta["local_timestamp"]
            if name in prev_us and us < prev_us[name] - 2 ** 31:
                wraps[name] += 1
            prev_us[name] = us
            esp_ms = (us + wraps[name] * 2 ** 32) / 1000.0
            anchor.setdefault(name, host_ts * 1000.0 - esp_ms)              # horloge ESP calée sur le PC
            t_ms = int(round(esp_ms + anchor[name]))
            v = lltf_standard_order(fr.raw, fr.meta.get("secondary_channel", 0))
            rows[name].append([t_ms] + [int(x) for x in v])
            clf.push(name, t_ms, v)

    print("=" * 70)
    print("Session guidée : %d consignes, %.0f s + %.0f s d'attente. Sortie : %s" % (len(script), total, a.attente, out_dir))
    print("Modèle :", a.ckpt)
    print("Écoute UDP :%d ... vérifiez que les 4 récepteurs envoient (compteurs ci-dessous)." % a.udp)
    print("=" * 70)
    preds = []
    t_start = time.time() + a.attente
    seg_bounds = np.cumsum([0] + [d for _, d in script])
    last_step, cur_seg = 0.0, -1
    try:
        while True:
            drain()
            now = time.time()
            rel = now - t_start
            if rel >= total:
                break
            k = int(np.searchsorted(seg_bounds, rel, side="right") - 1) if rel >= 0 else -1
            if k != cur_seg and k >= 0:
                cur_seg = k
                st, d = script[k]
                beep("go")
                print("\n\n>>> [%5.1f s] MAINTENANT : %s   (%d s)\n" % (rel, CONSIGNES[st], d))
                nxt = script[k + 1][0] if k + 1 < len(script) else None
                if nxt:
                    print("    ensuite : %s" % CONSIGNES[nxt])
            if now - last_step >= 0.1:
                last_step = now
                r = clf.step(int(now * 1000))
                if r is not None:
                    r["t_session"] = round(rel, 2)
                    preds.append(r)
                    consigne = script[cur_seg][0] if cur_seg >= 0 else "préparation"
                    rest = (seg_bounds[cur_seg + 1] - rel) if cur_seg >= 0 else -rel
                    if cur_seg >= 0 and 0 < rest <= 3 and abs(rest - round(rest)) < 0.06:
                        beep("tick")
                    ev = r["evenement"]
                    sys.stdout.write("\r[%6.1f s] consigne: %-8s (reste %4.1f s) | modèle: %-8s %3d%% | trames %s %s" % (
                        rel, consigne, max(rest, 0), r["etat"].upper(), 100 * r["probs"].get(r["etat"], 0),
                        " ".join("%s:%d" % (n[-1], v) for n, v in r["trames"].items()),
                        ("  <<< %s" % ("ALERTE CHUTE" if ev["type"] == "alerte_chute" else ev["vers"].upper())) if ev else ""))
                    sys.stdout.flush()
                    if ev:
                        print()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrompu : on enregistre ce qui a été reçu.")
    stop.set()
    drain()

    # ------------------------------------------------------------- enregistrement
    t0_ms = int(t_start * 1000)
    end_s = min(time.time() - t_start, total)
    for n in cfg.RX_NAMES:
        r = sorted(rows.get(n, []))
        with open(os.path.join(out_dir, n + ".csv"), "w", newline="") as f:
            csv.writer(f).writerows(r)
    segs = [[round(float(seg_bounds[i]), 2), round(float(min(seg_bounds[i + 1], end_s)), 2), s]
            for i, (s, _) in enumerate(script) if seg_bounds[i] < end_s]
    meta = {"t0_ms": t0_ms, "consignes": segs, "reaction_s": REACTION_S, "lieu": a.lieu, "personne": a.personne,
            "protocole": a.scenario or a.protocole, "modele": os.path.basename(a.ckpt), "carte_udp": names,
            "trames": {n: len(rows.get(n, [])) for n in cfg.RX_NAMES},
            "remarque": "étiquettes = consignes ; ignorer REACTION_S après chaque changement ; "
                        "'chute' = chute puis au sol"}
    with open(os.path.join(out_dir, "consignes.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "etat"] + cfg.GROUPS)
        for p in preds:
            w.writerow([p["t_session"], p["etat"]] + [p["probs"][g] for g in cfg.GROUPS])

    # ------------------------------------------------------------- bilan
    ok = n = 0
    per = collections.defaultdict(lambda: [0, 0])
    for p in preds:
        t = p["t_session"]
        if t < 0 or p["etat"] == "analyse":
            continue
        seg = next((s for s in segs if s[0] <= t < s[1]), None)
        if seg is None or t - seg[0] < REACTION_S:
            continue
        good = expected_ok(p["etat"], seg[2])
        ok += good
        n += 1
        per[seg[2]][0] += good
        per[seg[2]][1] += 1
    fall_seg = next((s for s in segs if s[2] == "chute"), None)
    alerts = [p["t_session"] for p in preds if p["evenement"] and p["evenement"]["type"] == "alerte_chute"]
    false_alerts = [t for t in alerts if fall_seg is None or t < fall_seg[0] - 0.5]
    bilan = {"accord": round(ok / max(n, 1), 3), "par_etat": {k: round(v[0] / max(v[1], 1), 3) for k, v in per.items()},
             "chute_consigne_s": fall_seg[0] if fall_seg else None,
             "chute_alerte_s": next((t for t in alerts if fall_seg and t >= fall_seg[0] - 0.5), None),
             "fausses_alertes_s": false_alerts, "trames": meta["trames"],
             "trames_par_s": {k: round(v / max(end_s + a.attente, 1), 1) for k, v in meta["trames"].items()}}
    with open(os.path.join(out_dir, "bilan.json"), "w", encoding="utf-8") as f:
        json.dump(bilan, f, indent=1, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("BILAN :", json.dumps(bilan, ensure_ascii=False))
    print("Session enregistrée :", out_dir)
    if min(meta["trames"].values()) < 100:
        print("ATTENTION : un récepteur n'a presque rien envoyé (vérifier --map / alimentation / Wi-Fi).")


if __name__ == "__main__":
    main()
