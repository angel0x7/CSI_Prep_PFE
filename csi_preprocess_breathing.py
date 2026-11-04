#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prétraitement CSI et estimation expérimentale de fréquence cardiaque."""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from csi_preprocess import Config, collect_inputs, process_file

log = logging.getLogger("csi_hr")


def estimate_windows(
    data: np.ndarray,
    times: np.ndarray,
    fs: float,
    low: float,
    high: float,
    window_s: float,
    step_s: float,
    subcarriers: np.ndarray,
    selected_subcarrier: int | None,
) -> list[dict]:
    """Estime le pic spectral de chaque fenêtre, en combinant les PSD des canaux."""
    window_size = int(round(window_s * fs))
    step_size = int(round(step_s * fs))
    if window_size < 2 or step_size < 1:
        raise ValueError("la fenêtre et le pas doivent représenter au moins un échantillon")
    if len(data) < window_size:
        raise ValueError(
            f"signal trop court ({len(data) / fs:.1f} s) pour une fenêtre de {window_s:.1f} s"
        )

    if selected_subcarrier is not None:
        matches = np.flatnonzero(subcarriers == selected_subcarrier)
        if not len(matches):
            raise ValueError(
                f"sous-porteuse {selected_subcarrier} absente des sous-porteuses conservées"
            )
        channel_indices = matches
    else:
        channel_indices = np.arange(data.shape[1])

    results = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        segment = signal.detrend(data[start:end, channel_indices], axis=0, type="constant")
        freqs, psd = signal.welch(
            segment,
            fs=fs,
            axis=0,
            nperseg=window_size,
            detrend="constant",
            scaling="density",
        )
        band = (freqs >= low) & (freqs <= high)
        if not band.any():
            raise ValueError("aucun point spectral dans la bande demandée")

        band_psd = psd[band]
        if selected_subcarrier is None:
            channel_power = band_psd.sum(axis=0)
            normalized = band_psd / np.maximum(channel_power, np.finfo(float).eps)
            combined_psd = normalized.mean(axis=1)
        else:
            combined_psd = band_psd[:, 0]

        peak_idx = int(np.argmax(combined_psd))
        peak_freq = float(freqs[band][peak_idx])
        results.append({
            "window_start_s": float(times[start]),
            "window_end_s": float(times[end - 1]),
            "peak_hz": peak_freq,
            "hr_bpm": peak_freq * 60.0,
            "peak_power": float(combined_psd[peak_idx]),
            "channels_used": len(channel_indices),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Nettoie des CSI ESP32 et estime une fréquence cardiaque par fenêtres."
    )
    parser.add_argument("inputs", nargs="+", help="fichier(s) CSV ou dossier(s) à parcourir")
    parser.add_argument("-o", "--out", default="csi_hr", help="dossier de sortie")
    parser.add_argument("--pattern", default="*.csv", help="motif des fichiers dans les dossiers")
    parser.add_argument("--fs", type=float, default=50.0, help="fréquence de ré-échantillonnage (Hz)")
    parser.add_argument("--low", type=float, default=0.8, help="borne basse de recherche (Hz)")
    parser.add_argument("--high", type=float, default=2.5, help="borne haute de recherche (Hz)")
    parser.add_argument("--order", type=int, default=4, help="ordre du filtre Butterworth")
    parser.add_argument("--keep-sc", default="2-26,42-63", help="sous-porteuses conservées")
    parser.add_argument("--window", type=float, default=10.0, help="durée de fenêtre (secondes)")
    parser.add_argument("--step", type=float, default=1.0, help="pas entre fenêtres (secondes)")
    parser.add_argument("--signal", choices=("amp", "phase"), default="amp",
                        help="caractéristique analysée")
    parser.add_argument("--subcarrier", type=int, help="analyser uniquement cet indice ESP32 (0-63)")
    parser.add_argument("--no-hampel", action="store_true", help="désactiver le filtre de Hampel")
    parser.add_argument("--no-header-check", action="store_true",
                        help="désactiver le rejet des lignes à en-tête anormal")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.fs <= 0 or args.low <= 0 or args.low >= args.high or args.window <= 0 or args.step <= 0:
        parser.error("fs, bornes de fréquence et fenêtre/pas doivent être positifs, avec low < high")
    if args.high >= args.fs / 2:
        parser.error("--high doit être inférieur à la fréquence de Nyquist (fs/2)")
    if args.subcarrier is not None and not 0 <= args.subcarrier <= 63:
        parser.error("--subcarrier doit être compris entre 0 et 63")

    files = collect_inputs(args.inputs, args.pattern)
    if not files:
        log.error("aucun fichier à traiter")
        return 1

    cfg = Config(
        fs=args.fs,
        low=args.low,
        high=args.high,
        order=args.order,
        keep_sc=args.keep_sc,
        normalize="zscore",
        phase=args.signal == "phase",
        use_hampel=not args.no_hampel,
        header_check=not args.no_header_check,
    )
    out_dir = Path(args.out)
    summaries = []
    errors = []

    for src, root in files:
        try:
            result, source_summary, _ = process_file(src, cfg)
            feature_name = "phase" if args.signal == "phase" else "amp"
            estimates = estimate_windows(
                result[feature_name],
                result["t"],
                args.fs,
                args.low,
                args.high,
                args.window,
                args.step,
                result["subcarriers"],
                args.subcarrier,
            )
            rel = src.relative_to(root) if root != src else Path(src.name)
            output_path = out_dir / rel.parent / f"{rel.stem}_hr.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(estimates[0]))
                writer.writeheader()
                writer.writerows(estimates)

            summaries.append({
                "file": str(src),
                "output": str(output_path),
                "n_windows": len(estimates),
                "median_hr_bpm": float(np.median([row["hr_bpm"] for row in estimates])),
                "n_packets": source_summary["n_packets"],
                "n_rejected_packets": source_summary["n_rejected_packets"],
            })
            log.info("%s -> %s (%d fenêtres, médiane %.1f BPM)",
                     src, output_path, len(estimates), summaries[-1]["median_hr_bpm"])
        except Exception as exc:
            errors.append(f"{src}: {type(exc).__name__}: {exc}")

    if summaries:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        log.info("Résumé : %s", summary_path)
    for error in errors:
        log.error("échec -> %s", error)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
