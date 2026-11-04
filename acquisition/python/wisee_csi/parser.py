# -*- coding: utf-8 -*-
"""
Parseur des lignes CSI produites par le firmware officiel espressif/esp-csi
(examples/get-started/csi_recv).

Deux formats existent (cf. README esp-csi) :
  * ESP32 / ESP32-S3 / ESP32-C3 : 25 champs
      type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
      aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
      secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data
  * ESP32-C5 / C6 / C61 : 15 champs
      type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,
      local_timestamp,sig_len,rx_state,len,first_word,data

Le champ `data` est un tableau "[...]" d'entiers : pour chaque sous-porteuse,
la partie IMAGINAIRE puis la partie REELLE (ordre des LTF : LLTF, HT-LTF, STBC-HT-LTF).
"""

import csv
import io

import numpy as np

FIELDS_S3 = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
             "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
             "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
             "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

FIELDS_C6 = ["type", "id", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain",
             "channel", "local_timestamp", "sig_len", "rx_state", "len", "first_word", "data"]

LLTF_SUBCARRIERS = 64          # le LLTF (legacy) contient toujours 64 sous-porteuses


class CSIFrame(object):
    """Une mesure CSI decodee."""

    __slots__ = ("meta", "raw", "csi", "host_ts")

    def __init__(self, meta, raw, host_ts=None):
        self.meta = meta                      # dict des metadonnees (int quand possible)
        self.raw = raw                        # np.int16 : valeurs brutes [im0, re0, im1, re1, ...]
        self.csi = raw[1::2].astype(np.float32) + 1j * raw[0::2].astype(np.float32)
        self.host_ts = host_ts                # horodatage cote PC (s), ajoute par l'enregistreur

    @property
    def seq(self):
        return self.meta["id"]

    @property
    def amplitude(self):
        return np.abs(self.csi)

    @property
    def lltf(self):
        """Les 64 sous-porteuses du LLTF (presentes dans toutes les trames)."""
        return self.csi[:LLTF_SUBCARRIERS]


def _to_int(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_line(line, host_ts=None):
    """Decode une ligne 'CSI_DATA,...'. Renvoie un CSIFrame, ou None si la ligne
    n'est pas une ligne CSI valide (log ESP-IDF, en-tete, ligne tronquee...)."""
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None
    try:
        fields = next(csv.reader(io.StringIO(line)))
    except (csv.Error, StopIteration):
        return None

    if len(fields) == len(FIELDS_S3):
        names = FIELDS_S3
    elif len(fields) == len(FIELDS_C6):
        names = FIELDS_C6
    else:
        return None                                   # ligne tronquee ou corrompue

    meta = {k: _to_int(v) for k, v in zip(names[:-1], fields[:-1])}
    data = fields[-1].strip()
    if not (data.startswith("[") and data.endswith("]")):
        return None
    try:
        raw = np.array([int(x) for x in data[1:-1].split(",") if x.strip() != ""],
                       dtype=np.int16)
    except ValueError:
        return None
    # controle de coherence : 'len' annonce le nombre de valeurs du tableau
    if isinstance(meta.get("len"), int) and meta["len"] != raw.size:
        return None
    if raw.size < 2 or raw.size % 2:
        return None
    return CSIFrame(meta, raw, host_ts)


def format_line_s3(meta, raw):
    """Fabrique une ligne au format ESP32-S3 (utilise par le simulateur et les tests)."""
    head = ",".join(str(meta[k]) for k in FIELDS_S3[1:-3])
    body = ",".join(str(int(v)) for v in raw)
    return 'CSI_DATA,%s,%d,%d,"[%s]"' % (head, raw.size, meta.get("first_word", 0), body)
