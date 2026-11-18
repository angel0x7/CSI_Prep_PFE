# -*- coding: utf-8 -*-
"""
Interface graphique de test du modèle squelette (serveur web local, aucune dépendance en plus).

    python -m skeleton.viewer            -> ouvre http://127.0.0.1:8765 dans le navigateur
    python -m skeleton.viewer --port 9000 --no-browser

Jeux de données proposés (s'ils existent) :
  DATASET        vraies prises (étiquette = activité de la prise, instant de chute estimé)
  DATASET_AUG    vraies prises + prises augmentées
  DATASET_SYNTH  prises simulées, avec vérité terrain squelette et position (comparaison + erreur en mm)
"""
import argparse
import gzip
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from . import config as cfg
from . import labels as L
from .features import load_take, take_features
from .infer import CLIP_CKPT, DEFAULT_CKPT, infer_take, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "acquisition", "data"))
DATASETS = ["DATASET", "DATASET_AUG", "DATASET_SYNTH"]
STATIC = os.path.join(HERE, "viewer_static")


class Backend(object):
    def __init__(self):
        self.models = {"temps_reel": load_model(DEFAULT_CKPT)}
        if os.path.exists(CLIP_CKPT):
            self.models["prise_entiere"] = load_model(CLIP_CKPT)
        from .live import HOLDOUT_CKPT
        self.live_model_name = "modèle de test (entraîné sans les prises mises de côté)"
        if os.path.exists(HOLDOUT_CKPT):
            self.live_model = load_model(HOLDOUT_CKPT)
        else:
            self.live_model = self.models["temps_reel"]
            self.live_model_name = "modèle principal (ATTENTION : a vu toutes les prises réelles)"
        if os.path.exists(HOLDOUT_CKPT):
            self.models["test_honnete"] = self.live_model
        self.cache = {}
        self.lock = threading.Lock()
        self.video_lock = threading.Lock()

    def mode(self, m):
        if m in self.models:
            return m
        return "prise_entiere" if "prise_entiere" in self.models else "temps_reel"

    # ------------------------------------------------------------------ liste
    def takes(self):
        out = {}
        for ds in DATASETS:
            p = os.path.join(DATA, ds, "index.csv")
            if not os.path.exists(p):
                continue
            idx = pd.read_csv(p)
            rows = []
            for _, r in idx.iterrows():
                if not os.path.isdir(os.path.join(DATA, ds, r.chemin)):
                    continue
                q = str(r.get("qualite", "")) if "qualite" in idx.columns else ""
                if ds == "DATASET_SYNTH":
                    q = "simulee"
                rows.append({"path": r.chemin, "activity": r.activite, "persons": int(r.personnes),
                             "quality": q, "augmented": bool(r.get("augmentee", False)) if "augmentee" in idx.columns else False})
            out[ds] = rows
        try:
            from .holdout import holdout_takes
            held = sorted(t for v in holdout_takes().values() for t in v)
        except FileNotFoundError:
            held = []
        return {"datasets": out, "groups": cfg.GROUPS, "modes": list(self.models), "holdout": held,
                "room": {"L": cfg.ROOM_L, "W": cfg.ROOM_W, "rx": cfg.RX_POS, "tx": cfg.TX_POS}}

    # ------------------------------------------------------------------ prédiction
    def take_dir(self, ds, path):
        d = os.path.normpath(os.path.join(DATA, ds, path))
        if ds not in DATASETS or not d.startswith(os.path.join(DATA, ds) + os.sep) or not os.path.isdir(d):
            raise FileNotFoundError(path)
        return d

    def predict(self, ds, path, mode=None):
        mode = self.mode(mode)
        key = (ds, path, mode)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        d = self.take_dir(ds, path)
        res = infer_take(d, self.models[mode])
        rx = load_take(d)
        X, t, info = take_features(rx)
        res["energy"], res["received"] = self._energy(X)
        act = path.split("/")[0]
        res["activity_true"] = act
        parts = path.split("/")
        res["persons"] = int(parts[1][1:]) if len(parts) > 1 and parts[1][1:].isdigit() else None
        res["truth"] = self._truth(ds, d, act, X, t, info)
        res["note"] = self._note(ds, d)
        for fr in res["frames"]:                    # allège la réponse
            fr.pop("keypoints_coco", None)
        with self.lock:
            self.cache[key] = res
        return res

    def results(self):
        """Tous les résultats de test enregistrés (skeleton/results/*.json), avec leur date de calcul."""
        import datetime
        rdir = os.path.join(HERE, "results")
        out = {"cv": {}, "autres": {}}
        for f in sorted(os.listdir(rdir)):
            if not f.endswith(".json"):
                continue
            path = os.path.join(rdir, f)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            stamp = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d/%m/%Y %H:%M")
            item = {"fichier": "skeleton/results/" + f, "date": stamp, "data": data}
            if f.startswith("cv_"):
                out["cv"][f[3:-5]] = item
            else:
                out["autres"][f[:-5]] = item
        return out

    def presets(self):
        from .sequences import PRESETS, STATES
        info = {"presets": PRESETS, "states": STATES, "model": self.live_model_name}
        try:
            from .holdout import holdout_takes
            info["holdout"] = {k: len(v) for k, v in holdout_takes().items()}
        except FileNotFoundError:
            info["holdout"] = None
        return info

    def live(self, scenario, source, speed, seed, stop):
        """Générateur d'événements du test en direct (rejeu au rythme réel)."""
        from .live import LiveClassifier, replay, score
        from .sequences import build
        seq = build(scenario, source, seed)
        tk = seq.get("truth_kp")
        yield {"type": "init", "truth": seq["truth"], "duration": round(seq["duration"], 2),
               "meta": seq["meta"], "model": self.live_model_name,
               "truth_kp": None if tk is None else np.round(tk.astype(float), 3).tolist(),
               "warmup_s": round((self.live_model.receptive_field - 1) / cfg.FS, 2)}
        clf = LiveClassifier(self.live_model)
        results = []
        for r in replay(seq, clf, speed=speed, stop=stop):
            results.append(r)
            r["type"] = "tick"
            yield r
        yield {"type": "fin", "score": score(results, seq["truth"])}

    def video(self, ds, path, mode=None):
        """GIF de la prise (rendu matplotlib, mis en cache dans skeleton/cache/videos)."""
        from .video import render
        mode = self.mode(mode)
        res = self.predict(ds, path, mode)
        vdir = os.path.join(HERE, "cache", "videos")
        os.makedirs(vdir, exist_ok=True)
        out = os.path.join(vdir, "%s__%s__%s.gif" % (ds, path.replace("/", "_"), mode))
        with self.video_lock:
            if not os.path.exists(out):
                full = dict(res, frames=[dict(f) for f in res["frames"]])
                render(full, out)
        with open(out, "rb") as f:
            return f.read(), os.path.basename(out)

    @staticmethod
    def _energy(X):
        nb, fp = cfg.N_BANDS, cfg.FEAT_PER_RX
        e, m = [], []
        for k in range(len(cfg.RX_NAMES)):
            F = X[:, k * fp:(k + 1) * fp]
            e.append(np.round(np.sqrt((F[:, :nb] ** 2).mean(1)), 3).tolist())
            m.append(F[:, -1].astype(int).tolist())
        return e, m

    @staticmethod
    def _note(ds, d):
        try:
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                s = json.load(f)
        except OSError:
            return {}
        out = {"note": s.get("note", "")}
        rec = s.get("receivers", {})
        out["loss_pct"] = {k: v.get("loss_pct") for k, v in rec.items() if "loss_pct" in v}
        out["frames"] = {k: v.get("frames") for k, v in rec.items()}
        if "augmentation" in s:
            out["augmentation"] = s["augmentation"]
        if "simulation" in s:
            out["simulation"] = s["simulation"]
        return out

    @staticmethod
    def _truth(ds, d, act, X, t, info):
        """Vérité terrain alignée sur les bins : groupe (+ keypoints, position pour le simulé)."""
        G = {g: i for i, g in enumerate(cfg.GROUPS)}
        lab_path = os.path.join(d, "labels.npz")
        if os.path.exists(lab_path):
            lab = np.load(lab_path)
            off = (info["t_start_ms"] - int(lab["t0_ms"])) / 1000.0
            i = np.clip(np.round((t + 0.5 / cfg.FS + off) * cfg.TX_RATE_HZ).astype(int), 0, len(lab["t"]) - 1)
            return {"kind": "simulation", "group": lab["group"][i].astype(int).tolist(),
                    "kp": np.round(lab["kp_body"][i].astype(float), 3).tolist(),
                    "pos": np.round(lab["pos"][i].astype(float), 3).tolist(),
                    "heading": np.round(lab["heading"][i].astype(float), 3).tolist()}
        if act not in G:
            return {"kind": "aucune"}
        tf = None
        if act == "chute":
            tf = L.fall_time(X)[0]
        y = L.frame_groups(act, len(X), tf, ignore=0.0)
        return {"kind": "faible (activité de la prise%s)" % (", chute estimée à %.2f s" % tf if tf else ""),
                "group": y.tolist(), "t_fall": tf}


def make_handler(backend):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            b = body if isinstance(body, bytes) else body.encode("utf-8")
            gz = ("gzip" in self.headers.get("Accept-Encoding", "") and len(b) > 1024
                  and not ctype.startswith("image/"))
            if gz:                                   # JSON ~10x plus petit
                b = gzip.compress(b, compresslevel=5)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "no-store")
            for k, v in getattr(self, "extra_headers", {}).items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            # envoi par morceaux : de gros write() uniques se bloquaient parfois vers 128 Ko sous Windows
            for k in range(0, len(b), 32768):
                self.wfile.write(b[k:k + 32768])
            self.wfile.flush()
            self.close_connection = True

        def _sse(self, gen):
            """Flux 'text/event-stream' : un message JSON par événement, envoyé au fil de l'eau."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            def event(obj):
                self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            try:
                for ev in gen:
                    event(ev)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self._stop.set()                             # l'utilisateur a arrêté / fermé la page
            except Exception as e:
                try:
                    event({"type": "erreur", "message": str(e)})
                except OSError:
                    pass
            self.close_connection = True

        def do_GET(self):
            self._stop = threading.Event()
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/takes":
                    return self._send(200, json.dumps(backend.takes()))
                if u.path == "/api/predict":
                    return self._send(200, json.dumps(backend.predict(q["ds"][0], q["take"][0],
                                                                      q.get("mode", [None])[0])))
                if u.path == "/api/results":
                    return self._send(200, json.dumps(backend.results(), ensure_ascii=False))
                if u.path == "/api/presets":
                    return self._send(200, json.dumps(backend.presets(), ensure_ascii=False))
                if u.path == "/api/live":
                    seed = q.get("seed", [""])[0]
                    return self._sse(backend.live(q["scenario"][0], q.get("source", ["simule"])[0],
                                                  float(q.get("speed", ["1"])[0]),
                                                  int(seed) if seed.strip().lstrip("-").isdigit() else None,
                                                  self._stop))
                if u.path == "/api/video":
                    data, name = backend.video(q["ds"][0], q["take"][0], q.get("mode", [None])[0])
                    self.extra_headers = {"Content-Disposition": 'attachment; filename="%s"' % name}
                    return self._send(200, data, "image/gif")
                name = "index.html" if u.path in ("/", "/index.html") else None
                if name:
                    with open(os.path.join(STATIC, name), "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                return self._send(404, '{"error":"introuvable"}')
            except FileNotFoundError as e:
                return self._send(404, json.dumps({"error": "prise introuvable : %s" % e}))
            except ValueError as e:
                if "aucun" in str(e):
                    return self._send(422, json.dumps({"error": "aucune trame reçue dans cette prise (inutilisable)"}))
                return self._send(500, json.dumps({"error": "ValueError: %s" % e}))
            except Exception as e:                                 # affiché dans l'interface
                return self._send(500, json.dumps({"error": "%s: %s" % (type(e).__name__, e)}))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    backend = Backend()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(backend))
    url = "http://127.0.0.1:%d" % a.port
    print("Interface squelette CSI : %s   (Ctrl+C pour arrêter)" % url)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
