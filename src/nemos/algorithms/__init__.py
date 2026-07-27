"""Bandit algorithms.

``NeMoS`` is our method (nearest-neighbor UCB with active learning); the
remaining classes are the baselines used in the paper.
"""

from nemos.algorithms.nemos import NeMoS
from nemos.algorithms.linucb import LinUCBBandit
from nemos.algorithms.neuronal_s_nets import ExploitationNet, ExplorationNet
from nemos.algorithms.pakucb import PAKUCB

__all__ = [
    "NeMoS",
    "LinUCBBandit",
    "ExploitationNet",
    "ExplorationNet",
    "PAKUCB",
]
