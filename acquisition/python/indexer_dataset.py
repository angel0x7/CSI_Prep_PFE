# -*- coding: utf-8 -*-
"""
WISEE - (re)construit data/DATASET/index.csv a partir des summary.json de chaque prise.

Arborescence attendue : data/DATASET/<activite>/p<N>/prise_<NNN>/summary.json
Qualite : ok (pertes <= 5 % sur tous les RX), degradee (> 5 % sur au moins un RX),
          inutilisable (au moins un RX avec moins de 100 trames).

Usage : python indexer_dataset.py                     (data/DATASET a cote du script)
        python indexer_dataset.py chemin/vers/DATASET
"""
import csv
import glob
import json
import os
import sys

RACINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "DATASET")
RX = ["RX1", "RX2", "RX3", "RX4"]


def main():
    global RACINE
    if len(sys.argv) > 1:
        RACINE = os.path.abspath(sys.argv[1])
    rows = []
    for f in sorted(glob.glob(os.path.join(RACINE, "*", "p*", "prise_*", "summary.json"))):
        rel = os.path.relpath(os.path.dirname(f), RACINE).replace("\\", "/")
        act, pn, prise = rel.split("/")
        j = json.load(open(f, encoding="utf-8"))
        r = j.get("receivers", {})
        if not r:
            continue
        frames = min(v["frames"] for v in r.values())
        lmax = max(v["loss_pct"] for v in r.values())
        q = "inutilisable" if frames < 100 else ("ok" if lmax <= 5 else "degradee")
        d = j.get("date", "")
        date = "%s-%s-%s %s:%s:%s" % (d[:4], d[4:6], d[6:8], d[9:11], d[11:13], d[13:15]) if len(d) == 15 else d
        anc = j.get("ancien_chemin", "").replace("data/DATASET/", "")
        rows.append([rel, act, pn[1:], prise, date, round(max(v["duration_s"] for v in r.values()), 1)]
                    + [r.get(k, {}).get("mean_fps", "") for k in RX]
                    + [r.get(k, {}).get("loss_pct", "") for k in RX]
                    + [lmax, j.get("sync", {}).get("common_pct_of_emitted", ""), q, anc])
    hdr = (["chemin", "activite", "personnes", "prise", "date", "duree_s"]
           + ["fps_%s" % k for k in RX] + ["pertes_%s_pct" % k for k in RX]
           + ["pertes_max_pct", "trames_communes_pct", "qualite", "ancien_chemin"])
    with open(os.path.join(RACINE, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    n = {q: sum(1 for x in rows if x[-2] == q) for q in ("ok", "degradee", "inutilisable")}
    print("%d prises indexees -> %s   %s" % (len(rows), os.path.join(RACINE, "index.csv"), n))


if __name__ == "__main__":
    main()
