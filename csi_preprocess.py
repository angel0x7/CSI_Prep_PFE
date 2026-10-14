#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csi_preprocess.py - Nettoyage + filtrage de données CSI ESP32 (Wi-Fi sensing)

Format d'entrée (un fichier .txt / .csv par enregistrement, une ligne par paquet) :
    timestamp_ms, v0, v1, ..., v127
où v = 64 paires (imaginaire, réel) -> 64 sous-porteuses (LLTF, ESP32).

Chaîne de traitement
--------------------
 1. Lecture robuste       : lignes tronquées / non numériques / timestamps dupliqués ignorés
 2. Nettoyage structurel  : en-tête constant (4 premières valeurs), bande de garde,
                            sous-porteuses nulles, lignes décalées (en-tête anormal)
 3. Amplitude (+ phase)   : |H| ; phase nettoyée (déroulée + retrait pente/offset linéaire)
 4. Filtre de Hampel      : suppression des valeurs aberrantes (impulsions) par sous-porteuse
 5. Ré-échantillonnage    : grille temporelle uniforme (interpolation linéaire, trous comblés)
 6. Filtre Butterworth    : passe-bande / passe-bas / passe-haut, phase nulle (sosfiltfilt)
 7. Normalisation (option): z-score ou min-max par sous-porteuse
 8. Export                : .npz (défaut) ou .csv  + summary.csv (+ figures de contrôle)

Exemples
--------
    # un fichier, préréglage "activité" (chute, marche, ...)
    python csi_preprocess.py csi_data_1720461632.txt -o out/ --mode activity --plot

    # tout un dossier (récursif), préréglage "respiration", 8 processus
    python csi_preprocess.py data/Datasets/Breathing* -o out_breath/ --mode breathing -j 8

    # paramètres manuels : passe-bande 0.5-8 Hz, z-score, phase incluse
    python csi_preprocess.py data/Datasets/Activity -o out/ --low 0.5 --high 8 \
           --normalize zscore --phase

Dépendances : numpy, scipy  (matplotlib optionnel pour --plot)
"""
from __future__ import annotations

import argparse
import logging

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

log = logging.getLogger("csi")

N_SC = 64                    # sous-porteuses LLTF ESP32
N_COLS = 1 + 2 * N_SC        # timestamp + 64 * (imag, real)

# Sous-porteuses conservées par défaut (indices 0..63 de l'ESP32, ordre 0..31 puis -32..-1) :
#   - 0, 1      : DC + 4 premiers octets invalides (valeurs constantes dans vos fichiers)
#   - 27..41    : bande de garde (nulle), dont la position dérive de ~27 à ~40 selon le paquet
# => 2-26 (fréquences +2..+26) et 42-63 (fréquences -22..-1) = 47 sous-porteuses fiables
DEFAULT_KEEP_SC = "2-26,42-63"

PRESETS = {
    # Fs = 50 Hz -> Nyquist = 25 Hz
    "activity":  dict(low=0.3, high=10.0),   # mouvements corps entier (chute, marche, assis...)
    "breathing": dict(low=0.1, high=0.6),    # 6-36 resp/min (10/15/20 BPM = 0.17/0.25/0.33 Hz)
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    fs: float = 50.0
    low: float = 0.3
    high: float = 10.0
    order: int = 4
    keep_sc: str = DEFAULT_KEEP_SC
    hampel_win: int = 11            # fenêtre (en paquets), impaire
    hampel_sigma: float = 5.0
    hampel_rel_floor: float = 0.10  # un point n'est aberrant que s'il dévie aussi de >10 % de la médiane
    use_hampel: bool = True
    header_check: bool = True
    max_bad_sc_frac: float = 0.30   # >30 % de sous-porteuses nulles -> paquet rejeté
    max_gap_s: float = 0.25         # trou temporel signalé dans le résumé
    normalize: str = "none"         # none | zscore | minmax
    phase: bool = False
    fmt: str = "npz"                # npz | csv
    plot: bool = False


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def parse_index_spec(spec: str) -> np.ndarray:
    """'2-26,42-63' -> array([2,...,26,42,...,63])"""
    idx: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            idx.extend(range(int(a), int(b) + 1))
        else:
            idx.append(int(part))
    idx_arr = np.array(sorted(set(idx)), dtype=int)
    if idx_arr.size == 0 or idx_arr.min() < 0 or idx_arr.max() >= N_SC:
        raise ValueError(f"Spécification de sous-porteuses invalide : {spec!r}")
    return idx_arr


def sc_freq_index(idx: np.ndarray) -> np.ndarray:
    """Indice tableau ESP32 (0..63) -> indice fréquentiel signé (-32..31)."""
    return np.where(idx < 32, idx, idx - N_SC)


def fill_nan_along_time(x: np.ndarray) -> np.ndarray:
    """Interpolation linéaire des NaN colonne par colonne (index = numéro de ligne)."""
    y = x.copy()
    t = np.arange(len(y))
    for k in range(y.shape[1]):
        m = np.isnan(y[:, k])
        if m.all():
            y[:, k] = 0.0
        elif m.any():
            y[m, k] = np.interp(t[m], t[~m], y[~m, k])
    return y


# --------------------------------------------------------------------------- #
# 1. Lecture
# --------------------------------------------------------------------------- #
def load_raw(path: Path):
    """Retourne (timestamps_ms [T], raw [T,128], stats)."""
    ts_list, rows = [], []
    n_total = n_bad = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            parts = line.split(",")
            if len(parts) != N_COLS:          # ligne tronquée / en-tête texte / etc.
                n_bad += 1
                continue
            try:
                arr = np.asarray(parts, dtype=np.float64)
            except ValueError:
                n_bad += 1
                continue
            if not np.isfinite(arr).all():
                n_bad += 1
                continue
            ts_list.append(arr[0])
            rows.append(arr[1:])
    if not rows:
        raise ValueError("aucune ligne valide")

    ts = np.asarray(ts_list)
    raw = np.vstack(rows)

    # tri chronologique + suppression des timestamps dupliqués
    order = np.argsort(ts, kind="stable")
    ts, raw = ts[order], raw[order]
    ts, uniq = np.unique(ts, return_index=True)
    n_dup = len(order) - len(uniq)
    raw = raw[uniq]

    stats = dict(n_lines=n_total, n_bad_lines=n_bad, n_dup_ts=n_dup)
    return ts, raw, stats


# --------------------------------------------------------------------------- #
# 2-3. Nettoyage structurel + amplitude / phase
# --------------------------------------------------------------------------- #
def header_outlier_rows(raw: np.ndarray, min_mode_frac: float = 0.9) -> np.ndarray:
    """
    Les 4 premières valeurs sont un en-tête quasi constant. Une ligne dont l'en-tête
    diffère du mode est le plus souvent décalée/corrompue.
    N'est appliqué que si le mode représente >= min_mode_frac des lignes.
    """
    heads = raw[:, :4]
    uniq, counts = np.unique(heads, axis=0, return_counts=True)
    mode = uniq[np.argmax(counts)]
    same = (heads == mode).all(axis=1)
    if same.mean() < min_mode_frac:
        return np.zeros(len(raw), dtype=bool)
    return ~same


def build_csi(raw: np.ndarray, keep_idx: np.ndarray, bad_rows: np.ndarray,
              max_bad_sc_frac: float):
    """
    raw [T,128] -> csi complexe [T,K] (NaN = invalide) et masque de lignes rejetées.
    Ordre des paires : (imaginaire, réel) comme indiqué dans le README du dataset.
    """
    imag = raw[:, 0::2]
    real = raw[:, 1::2]
    null = (imag == 0) & (real == 0)                       # sous-porteuse nulle
    csi = (real + 1j * imag)[:, keep_idx]
    null = null[:, keep_idx]
    csi[null] = np.nan

    row_bad = bad_rows | (null.mean(axis=1) > max_bad_sc_frac)
    csi[row_bad, :] = np.nan
    return csi, row_bad


def sanitize_phase(csi: np.ndarray, kfreq: np.ndarray) -> np.ndarray:
    """
    Phase brute ESP32 inexploitable (CFO/SFO/offset aléatoires par paquet).
    -> déroulement le long des sous-porteuses puis retrait d'une droite (pente + offset)
       ajustée par moindres carrés pour chaque paquet. Lignes avec NaN -> NaN.
    kfreq doit être trié par ordre croissant.
    """
    T, K = csi.shape
    ph = np.full((T, K), np.nan)
    ok = ~np.isnan(csi).any(axis=1)
    if not ok.any():
        return ph
    unwrapped = np.unwrap(np.angle(csi[ok]), axis=1)
    x = kfreq.astype(float)
    xc = x - x.mean()
    mean_ph = unwrapped.mean(axis=1, keepdims=True)
    slope = ((unwrapped - mean_ph) @ xc) / (xc @ xc)       # [n_ok]
    ph[ok] = unwrapped - mean_ph - slope[:, None] * xc[None, :]
    return ph


# --------------------------------------------------------------------------- #
# 4. Hampel
# --------------------------------------------------------------------------- #
def hampel_mask(x: np.ndarray, win: int, n_sigma: float, rel_floor: float) -> np.ndarray:
    """Masque booléen [T,K] des points aberrants (médiane glissante + MAD)."""
    if win % 2 == 0:
        win += 1
    y = fill_nan_along_time(x)
    med = median_filter(y, size=(win, 1), mode="nearest")
    dev = np.abs(y - med)
    mad = median_filter(dev, size=(win, 1), mode="nearest")
    thr = np.maximum(n_sigma * 1.4826 * mad, rel_floor * np.abs(med)) + 1e-9
    out = dev > thr
    out[np.isnan(x)] = False       # les NaN sont déjà invalides
    return out


# --------------------------------------------------------------------------- #
# 5. Ré-échantillonnage uniforme
# --------------------------------------------------------------------------- #
def resample_uniform(ts_ms: np.ndarray, x: np.ndarray, invalid: np.ndarray, fs: float):
    """Interpolation linéaire sur grille régulière ; les points invalides sont ignorés."""
    t = (ts_ms - ts_ms[0]) / 1000.0
    n = int(np.floor(t[-1] * fs)) + 1
    tn = np.arange(n) / fs
    out = np.zeros((n, x.shape[1]))
    n_dead = 0
    for k in range(x.shape[1]):
        v = ~invalid[:, k]
        if v.sum() < 2:
            n_dead += 1
            continue
        out[:, k] = np.interp(tn, t[v], x[v, k])
    return tn, out, n_dead


# --------------------------------------------------------------------------- #
# 6. Filtre Butterworth
# --------------------------------------------------------------------------- #
def design_filter(fs: float, low: float, high: float, order: int):
    nyq = fs / 2.0
    if high and high >= nyq:
        raise ValueError(f"--high ({high} Hz) doit être < Nyquist ({nyq} Hz)")
    if low and high:
        if low >= high:
            raise ValueError("--low doit être < --high")
        return signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    if high:
        return signal.butter(order, high / nyq, btype="lowpass", output="sos")
    if low:
        return signal.butter(order, low / nyq, btype="highpass", output="sos")
    return None


def apply_filter(sos, x: np.ndarray) -> np.ndarray:
    if sos is None:
        return x
    min_len = 3 * (2 * sos.shape[0] + 1) + 1
    if len(x) <= min_len * 3:
        raise ValueError(f"signal trop court ({len(x)} échantillons) pour le filtrage")
    return signal.sosfiltfilt(sos, x, axis=0)


# --------------------------------------------------------------------------- #
# 7. Normalisation
# --------------------------------------------------------------------------- #
def normalize(x: np.ndarray, how: str) -> np.ndarray:
    if how == "zscore":
        return (x - x.mean(0)) / (x.std(0) + 1e-12)
    if how == "minmax":
        mn, mx = x.min(0), x.max(0)
        return (x - mn) / (mx - mn + 1e-12)
    return x


# --------------------------------------------------------------------------- #
# Pipeline complet pour un fichier
# --------------------------------------------------------------------------- #
def _feature_pipeline(ts, feat, invalid, cfg: Config, sos, rel_floor: float):
    """Hampel -> ré-échantillonnage -> filtre. Retourne (tn, clean_unfiltered, filtered, stats)."""
    n_out = 0
    if cfg.use_hampel:
        outl = hampel_mask(feat, cfg.hampel_win, cfg.hampel_sigma, rel_floor)
        n_out = int(outl.sum())
        invalid = invalid | outl
    feat_f = np.where(invalid, 0.0, feat)
    tn, clean, n_dead = resample_uniform(ts, feat_f, invalid, cfg.fs)
    filt = apply_filter(sos, clean)
    filt = normalize(filt, cfg.normalize)
    return tn, clean, filt, n_out, n_dead


def process_file(path: Path, cfg: Config):
    ts, raw, stats = load_raw(path)

    keep = parse_index_spec(cfg.keep_sc)
    kf = sc_freq_index(keep)
    order = np.argsort(kf)                 # tri par fréquence croissante
    keep, kf = keep[order], kf[order]

    bad_rows = header_outlier_rows(raw) if cfg.header_check else np.zeros(len(raw), bool)
    csi, row_bad = build_csi(raw, keep, bad_rows, cfg.max_bad_sc_frac)

    amp = np.abs(csi)
    invalid = np.isnan(amp)
    sos = design_filter(cfg.fs, cfg.low, cfg.high, cfg.order)

    tn, amp_clean, amp_filt, n_out, n_dead = _feature_pipeline(
        ts, amp, invalid, cfg, sos, cfg.hampel_rel_floor)

    result = dict(
        t=tn, amp=amp_filt.astype(np.float32), amp_clean=amp_clean.astype(np.float32),
        subcarriers=keep, subcarrier_freq_idx=kf, fs=np.float64(cfg.fs),
    )

    if cfg.phase:
        ph = sanitize_phase(csi, kf)
        _, ph_clean, ph_filt, _, _ = _feature_pipeline(
            ts, ph, np.isnan(ph), cfg, sos, rel_floor=0.0)
        result["phase"] = ph_filt.astype(np.float32)
        result["phase_clean"] = ph_clean.astype(np.float32)

    dt = np.diff(ts)
    summary = dict(
        file=str(path),
        **stats,
        n_packets=len(ts),
        n_rejected_packets=int(row_bad.sum()),
        n_hampel_outliers=n_out,
        n_dead_subcarriers=n_dead,
        duration_s=round((ts[-1] - ts[0]) / 1000.0, 3),
        median_dt_ms=float(np.median(dt)),
        est_fs_hz=round(1000.0 / float(np.median(dt)), 2),
        n_gaps=int((dt > cfg.max_gap_s * 1000).sum()),
        n_out_samples=len(tn),
        n_subcarriers=len(keep),
    )
    # pour la figure : données brutes
    debug = dict(ts=ts, amp_raw=amp)
    return result, summary, debug


# --------------------------------------------------------------------------- #
# Export / figure
# --------------------------------------------------------------------------- #
def save_result(result: dict, out_path: Path, cfg: Config, src: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.fmt == "npz":
        np.savez_compressed(out_path.with_suffix(".npz"), source=str(src),
                            params=str(asdict(cfg)), **result)
    else:
        sc = result["subcarriers"]
        cols = ["t_s"] + [f"amp_sc{c}" for c in sc]
        data = np.column_stack([result["t"], result["amp"]])
        if "phase" in result:
            cols += [f"phase_sc{c}" for c in sc]
            data = np.column_stack([data, result["phase"]])
        np.savetxt(out_path.with_suffix(".csv"), data, delimiter=",",
                   header=",".join(cols), comments="", fmt="%.6g")


def make_plot(result: dict, debug: dict, out_png: Path, cfg: Config, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = debug["ts"][0]
    t_raw = (debug["ts"] - t0) / 1000.0
    k = result["amp"].shape[1] // 2                      # une sous-porteuse "milieu"
    sc = result["subcarriers"][k]

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(t_raw, debug["amp_raw"][:, k], lw=0.7, alpha=0.6, label="brut")
    ax[0].plot(result["t"], result["amp_clean"][:, k], lw=0.9, label="nettoyé (Hampel + rééchant.)")
    ax[0].set_title(f"{title} - amplitude, sous-porteuse {sc}")
    ax[0].set_xlabel("temps (s)"); ax[0].legend(loc="upper right")

    im = ax[1].imshow(result["amp"].T, aspect="auto", origin="lower",
                      extent=[result["t"][0], result["t"][-1], 0, result["amp"].shape[1]],
                      cmap="viridis")
    ax[1].set_title(f"amplitude filtrée ({cfg.low}-{cfg.high} Hz) - toutes sous-porteuses")
    ax[1].set_xlabel("temps (s)"); ax[1].set_ylabel("sous-porteuse (index)")
    fig.colorbar(im, ax=ax[1], pad=0.01)

    nper = min(1024, len(result["t"]))
    f1, p1 = signal.welch(result["amp_clean"] - result["amp_clean"].mean(0), cfg.fs, nperseg=nper, axis=0)
    f2, p2 = signal.welch(result["amp"] - result["amp"].mean(0), cfg.fs, nperseg=nper, axis=0)
    ax[2].semilogy(f1, p1.mean(1), label="avant filtre")
    ax[2].semilogy(f2, p2.mean(1), label="après filtre")
    ax[2].set_title("densité spectrale moyenne (Welch)")
    ax[2].set_xlabel("fréquence (Hz)"); ax[2].legend()
    ax[2].set_xlim(0, cfg.fs / 2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Traitement d'un fichier (fonction picklable pour le multiprocessing)
# --------------------------------------------------------------------------- #
def _job(args):
    src, root, out_dir, cfg = args
    try:
        result, summary, debug = process_file(src, cfg)
        rel = src.relative_to(root) if root != src else Path(src.name)
        out_path = out_dir / rel
        save_result(result, out_path, cfg, src)
        if cfg.plot:
            make_plot(result, debug, out_path.with_suffix(".png"), cfg, src.name)
        return summary, None
    except Exception as e:                 # un fichier défectueux ne doit pas tout arrêter
        return None, f"{src}: {type(e).__name__}: {e}"


def collect_inputs(inputs: list[str], pattern: str):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += [(f, p) for f in sorted(p.rglob(pattern))]
        elif p.is_file():
            files.append((p, p))
        else:
            log.warning("introuvable : %s", p)
    return files


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Nettoyage + filtrage de CSI ESP32.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="fichier(s) ou dossier(s) (recherche récursive)")
    ap.add_argument("-o", "--out", default="csi_clean", help="dossier de sortie")
    ap.add_argument("--pattern", default="csi_data_*.txt", help="motif de fichiers dans les dossiers")
    ap.add_argument("--mode", choices=list(PRESETS), default="activity",
                    help="préréglage de bande passante")
    ap.add_argument("--fs", type=float, default=50.0, help="fréquence de ré-échantillonnage (Hz)")
    ap.add_argument("--low", type=float, default=None, help="coupure basse (Hz), 0 = désactivée")
    ap.add_argument("--high", type=float, default=None, help="coupure haute (Hz), 0 = désactivée")
    ap.add_argument("--order", type=int, default=4, help="ordre Butterworth")
    ap.add_argument("--keep-sc", default=DEFAULT_KEEP_SC, help="sous-porteuses conservées (0-63)")
    ap.add_argument("--hampel-win", type=int, default=11, help="fenêtre Hampel (paquets)")
    ap.add_argument("--hampel-sigma", type=float, default=5.0, help="seuil Hampel (en sigma)")
    ap.add_argument("--no-hampel", action="store_true", help="désactive le filtre de Hampel")
    ap.add_argument("--no-header-check", action="store_true",
                    help="ne rejette pas les lignes à en-tête anormal")
    ap.add_argument("--normalize", choices=["none", "zscore", "minmax"], default="none")
    ap.add_argument("--phase", action="store_true", help="exporte aussi la phase nettoyée")
    ap.add_argument("--format", choices=["npz", "csv"], default="npz")
    ap.add_argument("--plot", action="store_true", help="figure de contrôle par fichier (PNG)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="processus parallèles")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    preset = PRESETS[a.mode]
    cfg = Config(
        fs=a.fs,
        low=preset["low"] if a.low is None else a.low,
        high=preset["high"] if a.high is None else a.high,
        order=a.order, keep_sc=a.keep_sc,
        hampel_win=a.hampel_win, hampel_sigma=a.hampel_sigma, use_hampel=not a.no_hampel,
        header_check=not a.no_header_check, normalize=a.normalize,
        phase=a.phase, fmt=a.format, plot=a.plot,
    )
    design_filter(cfg.fs, cfg.low, cfg.high, cfg.order)    # valide les paramètres tôt
    parse_index_spec(cfg.keep_sc)

    files = collect_inputs(a.inputs, a.pattern)
    if not files:
        log.error("aucun fichier à traiter")
        return 1
    out_dir = Path(a.out)
    log.info("%d fichier(s) | fs=%.0f Hz | filtre %.2f-%.2f Hz (ordre %d) | sortie : %s",
             len(files), cfg.fs, cfg.low, cfg.high, cfg.order, out_dir)

    jobs = [(f, root, out_dir, cfg) for f, root in files]
    summaries, errors = [], []
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = ex.map(_job, jobs, chunksize=4)
            for i, (s, e) in enumerate(results, 1):
                (summaries if s else errors).append(s or e)
                if i % 50 == 0:
                    log.info("%d/%d", i, len(jobs))
    else:
        for i, j in enumerate(jobs, 1):
            s, e = _job(j)
            (summaries if s else errors).append(s or e)
            log.debug("%d/%d %s", i, len(jobs), j[0].name)

    if summaries:
        import csv
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        tot_rej = sum(s["n_rejected_packets"] for s in summaries)
        tot_out = sum(s["n_hampel_outliers"] for s in summaries)
        tot_pk = sum(s["n_packets"] for s in summaries)
        log.info("OK : %d fichier(s), %d paquets, %d rejetés, %d points aberrants corrigés",
                 len(summaries), tot_pk, tot_rej, tot_out)
        log.info("résumé : %s", out_dir / "summary.csv")
    for e in errors:
        log.error("échec -> %s", e)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
