# -*- coding: utf-8 -*-
"""
Générateur de squelettes COCO-17 (3D, en mètres) par cinématique directe simplifiée.

Repère "corps" : x = côté gauche de la personne, y = devant, z = haut,
origine = projection au sol du milieu des hanches. Le repère "monde" s'obtient
avec to_world() (cap + position fournis par la localisation / le tracking).

Proportions : tables anthropométriques classiques (Drillis & Contini), en fraction
de la taille H.
"""
import numpy as np

from . import config as cfg

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]                # liste COCO (indices 0-based)

D = np.deg2rad
ANKLE_Z = 0.039          # hauteur de la cheville (fraction de H)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _seg(angle_fwd, abd=0.0, side=0.0):
    """Direction unitaire 'vers le bas' tournée de angle_fwd vers l'avant
    (et de abd vers l'extérieur du côté side=+1 gauche / -1 droit)."""
    return np.array([side * np.sin(abd),
                     np.sin(angle_fwd) * np.cos(abd),
                     -np.cos(angle_fwd) * np.cos(abd)])


def skeleton(H=cfg.NOMINAL_HEIGHT, torso_pitch=0.0, hip=(0.0, 0.0), knee=(0.0, 0.0),
             shoulder=(0.0, 0.0), elbow=(0.15, 0.15), arm_abd=0.12, pelvis_z=None,
             body_pitch=0.0, pelvis_dz=0.0):
    """Squelette (17, 3) dans le repère corps.

    Angles en radians, positifs = flexion vers l'avant. hip/knee/shoulder/elbow : (gauche, droite).
    pelvis_z : hauteur imposée du bassin (assis) ; sinon le pied le plus bas touche le sol.
    body_pitch : rotation de tout le corps vers l'avant autour des pieds (chute / au sol).
    """
    hip_w, sh_w = 0.191 * H, 0.259 * H
    L_torso = (0.818 - 0.530) * H            # hanche -> épaules
    L_neck = 0.10 * H
    L_thigh, L_shin = 0.245 * H, 0.246 * H
    L_uarm, L_farm = 0.186 * H, 0.146 * H

    P = np.zeros(3)
    u = np.array([0.0, np.sin(torso_pitch), np.cos(torso_pitch)])   # axe du tronc
    neck = P + u * L_torso
    k = np.zeros((17, 3))
    for i, side in ((0, 1.0), (1, -1.0)):                           # 0 = gauche, 1 = droit
        h = P + np.array([side * hip_w / 2, 0, 0])
        kn = h + _seg(hip[i]) * L_thigh
        an = kn + _seg(hip[i] - knee[i]) * L_shin
        s = neck + np.array([side * sh_w / 2, 0, 0])
        e = s + _seg(shoulder[i], arm_abd, side) * L_uarm
        w = e + _seg(shoulder[i] + elbow[i], arm_abd, side) * L_farm
        k[11 + i], k[13 + i], k[15 + i] = h, kn, an
        k[5 + i], k[7 + i], k[9 + i] = s, e, w
    head = neck + u * L_neck
    fwd = np.array([0.0, np.cos(torso_pitch), -np.sin(torso_pitch)])
    k[0] = head + fwd * 0.10 * H * 0.6                               # nez
    k[1] = head + fwd * 0.045 * H + np.array([0.03 * H, 0, 0.02 * H])
    k[2] = head + fwd * 0.045 * H + np.array([-0.03 * H, 0, 0.02 * H])
    k[3] = head + np.array([0.045 * H, 0, 0])
    k[4] = head + np.array([-0.045 * H, 0, 0])

    if pelvis_z is None:
        k[:, 2] += ANKLE_Z * H - k[15:17, 2].min() + pelvis_dz
    else:
        k[:, 2] += pelvis_z
    if body_pitch:
        pivot = np.array([0.0, k[15:17, 1].mean(), 0.0])
        k = (k - pivot) @ _rot_x(-body_pitch).T + pivot
        k[:, 2] -= min(0.0, k[:, 2].min() - 0.08)                    # rien sous le sol
    return k


# --------------------------------------------------------------------------
# Poses par activité
# --------------------------------------------------------------------------
def pose_stand(H=cfg.NOMINAL_HEIGHT, rng=None):
    j = (lambda s: rng.normal(0, s)) if rng is not None else (lambda s: 0.0)
    return skeleton(H, torso_pitch=D(2) + j(D(2)), hip=(j(D(3)), j(D(3))),
                    shoulder=(D(3) + j(D(5)), D(3) + j(D(5))), elbow=(D(10) + j(D(8)), D(10) + j(D(8))))


def pose_sit(H=cfg.NOMINAL_HEIGHT, rng=None):
    j = (lambda s: rng.normal(0, s)) if rng is not None else (lambda s: 0.0)
    seat = 0.46 if rng is None else rng.uniform(0.42, 0.50)
    return skeleton(H, torso_pitch=D(8) + j(D(5)), hip=(D(88) + j(D(5)), D(88) + j(D(5))),
                    knee=(D(90) + j(D(8)), D(90) + j(D(8))),
                    shoulder=(D(35) + j(D(10)), D(35) + j(D(10))), elbow=(D(55) + j(D(10)), D(55) + j(D(10))),
                    pelvis_z=seat + 0.06)


def pose_walk(phase, H=cfg.NOMINAL_HEIGHT, stride=1.0):
    """phase en radians (0..2pi), stride = amplitude relative (0.7 lent .. 1.3 rapide)."""
    hips, knees, shs = [], [], []
    for off in (0.0, np.pi):                                         # gauche, droite déphasées
        p = phase + off
        hips.append(D(22) * stride * np.sin(p))
        knees.append(D(8) + D(52) * stride * max(0.0, np.sin(p + 0.9)) ** 1.5)
        shs.append(-D(18) * stride * np.sin(p))
    return skeleton(H, torso_pitch=D(4), hip=hips, knee=knees, shoulder=shs,
                    elbow=(D(20), D(20)), pelvis_dz=0.02 * np.cos(2 * phase))


def pose_fall(alpha, H=cfg.NOMINAL_HEIGHT):
    """alpha = avancement de la chute (0 debout -> 1 au sol), chute vers l'avant du repère."""
    a = float(np.clip(alpha, 0, 1))
    knee = D(70) * np.sin(np.pi * a)                                 # genoux fléchis pendant la chute
    return skeleton(H, torso_pitch=D(25) * np.sin(np.pi * a), hip=(D(30) * np.sin(np.pi * a),) * 2,
                    knee=(knee, knee * 0.8), shoulder=(D(10) + D(110) * a, D(10) + D(100) * a),
                    elbow=(D(15),) * 2, body_pitch=D(88) * a ** 1.6)


def pose_ground(H=cfg.NOMINAL_HEIGHT, rng=None):
    """Allongé à plat après la chute. Les bras sont soit le long du corps, soit au-dessus
    de la tête : une flexion intermédiaire les ferait passer sous le sol, et la correction
    anti-pénétration soulèverait alors tout le corps (bug corrigé)."""
    j = (lambda s: rng.normal(0, s)) if rng is not None else (lambda s: 0.0)
    above = rng is not None and rng.random() < 0.5
    sh = D(172) if above else D(8)
    return skeleton(H, hip=(abs(j(D(4))), abs(j(D(4)))), knee=(D(5) + abs(j(D(6))), D(3)),
                    shoulder=(sh + j(D(6)), sh + j(D(6))), elbow=(D(8),) * 2,
                    body_pitch=D(90))


def atom_bank(H=cfg.NOMINAL_HEIGHT):
    """(N_ATOMS, 17, 3) : poses de référence du décodeur, dans l'ordre de cfg.ATOM_GROUP."""
    atoms = [np.zeros((17, 3)), pose_stand(H), pose_sit(H)]
    atoms += [pose_walk(2 * np.pi * i / cfg.N_WALK_ATOMS, H) for i in range(cfg.N_WALK_ATOMS)]
    atoms += [pose_fall((i + 0.5) / cfg.N_FALL_ATOMS, H) for i in range(cfg.N_FALL_ATOMS)]
    atoms += [pose_ground(H)]
    A = np.stack(atoms)
    assert len(A) == cfg.N_ATOMS
    return A


def walk_atom(phase):
    return 3 + int(np.round(phase / (2 * np.pi) * cfg.N_WALK_ATOMS)) % cfg.N_WALK_ATOMS


def fall_atom(alpha):
    return 3 + cfg.N_WALK_ATOMS + int(np.clip(alpha, 0, 0.999) * cfg.N_FALL_ATOMS)


def to_world(kp, x, y, heading):
    """Repère corps -> repère salle. heading = angle (rad) de l'avant du corps par rapport à +x."""
    return kp @ _rot_z(heading - np.pi / 2).T + np.array([x, y, 0.0])


def to_coco_2d(kp, img_w=192, img_h=256):
    """Projection de face (x latéral, z hauteur) en coordonnées image COCO [x, y, visibilité]."""
    s = img_h / 2.1
    out = np.zeros((len(kp), 3))
    out[:, 0] = img_w / 2 - kp[:, 0] * s
    out[:, 1] = img_h - 0.05 * img_h - kp[:, 2] * s
    out[:, 2] = 2
    return out
