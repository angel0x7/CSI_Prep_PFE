# -*- coding: utf-8 -*-
"""Tests unitaires du parseur (python -m unittest discover tests)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wisee_csi.parser import parse_line, format_line_s3, FIELDS_S3  # noqa: E402
from wisee_csi.metrics import StreamStats  # noqa: E402

# Ligne d'exemple copiee du README officiel esp-csi (format ESP32 / S3, len = 128)
README_LINE = ('CSI_DATA,0,94:d9:b3:80:8c:81,-30,11,1,6,1,0,1,0,1,0,0,-93,0,13,2,2751923,0,67,0,128,1,'
               '"[67,48,4,0,0,0,0,0,0,0,5,0,20,1,20,1,19,0,17,1,16,2,15,2,14,1,12,0,12,-1,12,-3,12,-4,'
               '13,-6,15,-7,16,-8,16,-8,16,-8,16,-6,15,-5,15,-4,14,-4,13,-4,12,-4,11,-4,10,-4,9,-5,8,-6,'
               '4,-4,8,-9,9,-10,9,-10,10,-11,11,-10,11,-10,12,-9,11,-8,11,-7,10,-6,9,-6,7,-6,6,-7,5,-7,'
               '5,-8,5,-9,5,-10,5,-11,5,-11,6,-11,7,-11,8,-11,9,-10,9,-9,8,-8,8,-7,1,-2,0,0,0,0,0,0,0,0]"')


class TestParser(unittest.TestCase):

    def test_readme_line(self):
        fr = parse_line(README_LINE)
        self.assertIsNotNone(fr)
        self.assertEqual(fr.meta["rssi"], -30)
        self.assertEqual(fr.meta["channel"], 13)
        self.assertEqual(fr.raw.size, 128)
        self.assertEqual(fr.csi.size, 64)
        # convention esp-csi : [imaginaire, reel] -> 2e sous-porteuse = 0 + 4j
        self.assertEqual(fr.csi[1], complex(0, 4))

    def test_rejects_garbage(self):
        self.assertIsNone(parse_line("I (1234) csi_recv: compensate_gain 1.0"))
        self.assertIsNone(parse_line(README_LINE[:150]))                 # ligne tronquee
        self.assertIsNone(parse_line(README_LINE.replace(",128,1,", ",130,1,")))  # len incoherent

    def test_roundtrip(self):
        raw = np.arange(-20, 20, dtype=np.int16)
        meta = {k: 0 for k in FIELDS_S3[1:-3]}
        meta["mac"] = "1a:00:00:00:00:00"
        meta["id"] = 42
        fr = parse_line(format_line_s3(meta, raw))
        self.assertEqual(fr.seq, 42)
        np.testing.assert_array_equal(fr.raw, raw)

    def test_loss_counter(self):
        st = StreamStats()
        raw = np.zeros(8, dtype=np.int16)
        meta = {k: 0 for k in FIELDS_S3[1:-3]}
        for i, seq in enumerate([0, 1, 2, 5, 5, 6]):                 # 3 et 4 perdus, 5 en double
            meta["id"] = seq
            st.update(parse_line(format_line_s3(meta, raw), host_ts=i * 0.01))
        self.assertEqual(st.n_lost, 2)
        self.assertEqual(st.n_dup, 1)


if __name__ == "__main__":
    unittest.main()
