# -*- coding: utf-8 -*-
"""python -m unittest discover skeleton/tests"""
import unittest

import numpy as np
import torch

from skeleton import config as cfg
from skeleton import poses as P
from skeleton import synth
from skeleton.features import sim_to_rx, take_features
from skeleton.labels import fall_time, frame_groups
from skeleton.model import SkeletonTCN


def bone_lengths(k):
    return np.array([np.linalg.norm(k[a] - k[b]) for a, b in P.SKELETON if a >= 5 and b >= 5])


class TestPoses(unittest.TestCase):
    def test_atoms_shape_and_rigid_bones(self):
        A = P.atom_bank()
        self.assertEqual(A.shape, (cfg.N_ATOMS, 17, 3))
        ref = bone_lengths(A[1])
        for a in A[1:]:                                    # longueurs d'os conservées
            np.testing.assert_allclose(bone_lengths(a), ref, rtol=1e-6)

    def test_feet_on_floor_and_head_up(self):
        st = P.pose_stand()
        self.assertAlmostEqual(st[15:17, 2].min(), P.ANKLE_Z * cfg.NOMINAL_HEIGHT, places=6)
        self.assertGreater(st[0, 2], 1.5)
        rng = np.random.default_rng(0)
        for g in [P.pose_ground()] + [P.pose_ground(1.75, rng) for _ in range(200)]:
            self.assertLess(g[:, 2].max(), 0.45)              # au sol : rien au-dessus de 45 cm
            self.assertGreaterEqual(g[:, 2].min(), 0.0)

    def test_to_world(self):
        k = P.to_world(P.pose_stand(), 2.0, 3.0, 0.0)
        np.testing.assert_allclose(k[11:13, :2].mean(0), [2.0, 3.0], atol=1e-6)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.env = synth.Environment(1)
        cls.rx, cls.lab, cls.info = synth.simulate_take("chute", rng, cls.env)

    def test_sim_format(self):
        for ts, raw, ids in self.rx.values():
            self.assertEqual(raw.shape[1], 128)
            self.assertTrue(np.all(np.diff(ts) > 0))
            self.assertTrue(np.all(raw[:, 2 * 27:2 * 38] == 0))   # sous-porteuses nulles 27..37
            self.assertTrue(np.all(raw[:, :2] == 0))               # DC

    def test_features(self):
        X, t, info = take_features(sim_to_rx(self.rx), t_start_ms=self.lab["t0_ms"] + 5, duration=cfg.TAKE_S)
        self.assertEqual(X.shape, (int(cfg.TAKE_S * cfg.FS), cfg.N_IN))
        self.assertTrue(np.isfinite(X).all())
        y = frame_groups("chute", len(X), fall_time(X)[0])
        self.assertIn(cfg.GROUPS.index("au_sol"), y)

    def test_missing_rx(self):
        rx = sim_to_rx(self.rx)
        del rx["RX3"]
        X, _, _ = take_features(rx, duration=cfg.TAKE_S)
        fp = cfg.FEAT_PER_RX
        self.assertTrue(np.all(X[:, 2 * fp:3 * fp] == 0))

    def test_model_shapes_and_causality(self):
        m = SkeletonTCN().eval()
        x = torch.randn(2, 60, cfg.N_IN)
        with torch.no_grad():
            o = m(x)
            x2 = x.clone()
            x2[:, 40:] += 5                                    # le futur ne doit pas changer le passé
            o2 = m(x2)
        self.assertEqual(o["kp"].shape, (2, 60, 17, 3))
        self.assertEqual(o["group_logp"].shape, (2, 60, len(cfg.GROUPS)))
        torch.testing.assert_close(o["kp"][:, :40], o2["kp"][:, :40])
        np.testing.assert_allclose(o["group_logp"].exp().sum(-1).numpy(), 1, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
