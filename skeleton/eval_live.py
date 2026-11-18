# -*- coding: utf-8 -*-
"""Évaluation en série du test en direct : chaque scénario prédéfini x source x graines,
rejoué en accéléré (même moteur que l'interface).   python -m skeleton.eval_live [-n 5]"""
import argparse
import json
import sys
import os

import numpy as np

from .infer import load_model
from .live import HOLDOUT_CKPT, LiveClassifier, replay, score
from .sequences import PRESETS, build


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # console Windows (cp1252)
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    a = ap.parse_args()
    model = load_model(HOLDOUT_CKPT)
    out = {}
    for src in ("reel", "simule"):
        for name, sc in PRESETS.items():
            rows = []
            for seed in range(a.n):
                seq = build(sc, src, seed)
                res = list(replay(seq, LiveClassifier(model), realtime=False))
                rows.append(score(res, seq["truth"]))
            fall = [r for r in rows if r["chute_verite_s"] is not None]
            d = [c["delai_s"] for r in rows for c in r["changements"] if c["delai_s"] is not None]
            miss = sum(c["delai_s"] is None for r in rows for c in r["changements"])
            out.setdefault(src, {})[name] = {
                "accord_moyen": round(float(np.mean([r["accord"] for r in rows])), 3),
                "chutes_alertees": "%d/%d" % (sum(r["chute_alerte_s"] is not None and not r["fausse_alerte"] for r in fall), len(fall)),
                "fausses_alertes": sum(r["fausse_alerte"] for r in rows),
                "changements_detectes": "%d/%d" % (len(d), len(d) + miss),
                "delai_median_s": None if not d else round(float(np.median(d)), 2)}
            print(src, "|", name, "|", out[src][name], flush=True)
    path = os.path.join(os.path.dirname(__file__), "results", "test_en_direct.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("->", path)


if __name__ == "__main__":
    main()
