# -*- coding: utf-8 -*-
"""
SkeletonTCN : CSI (4 RX) -> squelette COCO-17 3D.
  - mode temps réel (causal=True) : ne voit que le passé, pour le dashboard en direct ;
  - mode prise entière (causal=False, dilatations jusqu'à 32) : voit les 8 s d'un coup,
    passé et futur, comme une vidéo -> fiable dès t = 0.

    features (T, 216) ──► TCN causal (5 blocs dilatés, champ réceptif ~3 s)
                          ├─► logits sur 24 "atomes de pose" (debout, assis, 12 phases de marche,
                          │    8 étapes de chute, au sol, vide)
                          ├─► taille de la personne (1.55 – 1.95 m)
                          └─► résidu par keypoint (± 15 cm)
    keypoints = (softmax(atomes) · banque d'atomes) × taille/1.75 + résidu

Le décodeur par atomes garantit des squelettes anatomiquement plausibles même quand le
signal est faible, et la probabilité d'une activité = somme des probabilités de ses atomes.
~0.3 M paramètres : entraînement et inférence sur CPU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from .poses import atom_bank


class ChannelNorm(nn.Module):
    """LayerNorm sur les canaux, pas de temps par pas de temps (GroupNorm mélangerait
    le futur et le passé : interdit pour un modèle temps réel causal)."""

    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):                      # (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalBlock(nn.Module):
    """Bloc convolutif dilaté. causal=True : ne regarde que le passé (temps réel) ;
    causal=False : fenêtre centrée passé + futur (analyse d'une prise entière, comme une vidéo)."""

    def __init__(self, ch, dilation, k=3, p=0.1, causal=True):
        super().__init__()
        tot = (k - 1) * dilation
        self.pad = (tot, 0) if causal else (tot // 2, tot - tot // 2)
        self.c1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.n1 = ChannelNorm(ch)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.c1(F.pad(x, self.pad))
        h = F.gelu(self.n1(h))
        h = self.drop(self.c2(h))
        return F.gelu(x + h)


class SkeletonTCN(nn.Module):
    def __init__(self, n_in=cfg.N_IN, ch=96, dilations=(1, 2, 4, 8, 16), p=0.1, causal=True):
        super().__init__()
        self.causal = causal
        self.dilations = tuple(dilations)
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.register_buffer("atoms", torch.tensor(atom_bank(), dtype=torch.float32))   # (24,17,3)
        self.register_buffer("atom_group", torch.tensor(cfg.ATOM_GROUP_IDX, dtype=torch.long))
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.in_drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[CausalBlock(ch, d, p=p, causal=causal) for d in dilations])
        self.head_atom = nn.Conv1d(ch, cfg.N_ATOMS, 1)
        self.head_h = nn.Conv1d(ch, 1, 1)
        self.head_res = nn.Conv1d(ch, 17 * 3, 1)
        self.receptive_field = 1 + 2 * sum(dilations)

    def set_norm(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.as_tensor(np.maximum(sd, 1e-3), dtype=torch.float32))

    def forward(self, x):
        """x (B, T, N_IN) -> dict de sorties par pas de temps."""
        x = (x - self.mu) / self.sd
        h = self.in_drop(self.inp(x.transpose(1, 2)))
        h = self.blocks(h)
        logits = self.head_atom(h).transpose(1, 2)                          # (B,T,24)
        height = 1.55 + 0.4 * torch.sigmoid(self.head_h(h)).transpose(1, 2)  # (B,T,1)
        res = 0.15 * torch.tanh(self.head_res(h)).transpose(1, 2)           # (B,T,51)
        w = logits.softmax(-1)
        kp = torch.einsum("bta,akd->btkd", w, self.atoms) * (height[..., None] / cfg.NOMINAL_HEIGHT)
        present = 1 - w[..., 0:1, None]                                     # atome 0 = vide
        kp = kp + res.view(*res.shape[:2], 17, 3) * present
        logp = logits.log_softmax(-1)
        group_logp = torch.stack([torch.logsumexp(logp[..., self.atom_group == g], -1)
                                  for g in range(len(cfg.GROUPS))], -1)     # (B,T,6)
        return {"atom_logits": logits, "group_logp": group_logp, "kp": kp, "height": height[..., 0]}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
