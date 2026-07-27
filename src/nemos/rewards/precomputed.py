"""Replay of rewards computed offline — the protocol used for the paper's results.

`scripts/create_dataset/` writes one JSON file per (dataset, metric), holding every score
of every model on every prompt. :class:`PrecomputedRewards` draws from those scores instead
of generating images, which is what makes an experiment run in seconds on a CPU.
"""
import json
import random
from collections import defaultdict

import numpy as np
import torch

from nemos.rewards.base import RewardSource

# metric name -> (metadata file written by scripts/create_dataset, key holding the scores)
METRICS = {
    "clip": ("metadata.json", "clip_scores"),
    "imagereward": ("metadata_IR.json", "image_reward_scores"),
    "hpsv2": ("metadata_HPS.json", "hps_scores"),
}


def metadata_path(dataset_dir, metric):
    """Path of the metadata file holding `metric`'s scores for a dataset directory."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(METRICS)}")
    return f"{dataset_dir}/{METRICS[metric][0]}"


def load_score_map(path, models, metric):
    """Read a metadata file into `{prompt: {model: [scores]}}`.

    Entries whose scores are not a list of floats are skipped: `score_images.py` records
    `null` for images it could not load or score, and those must not reach the bandits.

    Returns:
        (scores_map, valid_prompts) where `valid_prompts` are the prompts for which every
        model in `models` has at least one score.
    """
    _, score_key = METRICS[metric]
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    scores_map = defaultdict(lambda: defaultdict(list))
    for e in raw:
        p, m, cs = e["prompt"], e["model"], e.get(score_key, [])
        if m in models and isinstance(cs, list) and all(isinstance(x, float) for x in cs):
            scores_map[p][m].extend(cs)

    valid = [p for p in scores_map if all(scores_map[p][m] for m in models)]
    return scores_map, valid


class PrecomputedRewards(RewardSource):
    """Rewards drawn from offline scores.

    For each model, `generations` of its stored scores are drawn without replacement and
    averaged, which is how a round's reward was defined for the paper: the mean score of a
    handful of that model's images for the prompt.

    Args:
        scores_map: `{prompt: {model: [scores]}}`, e.g. from :func:`load_score_map`. A
            scalar value instead of a list (the LLM-selection task stores one correctness
            value per question) is returned as-is, without consuming the RNG.
        generations: how many scores to average per model.
        device: torch device for the averaging.
        with_replacement: when a model has fewer than `generations` scores, draw with
            replacement (up to `generations` draws) instead of averaging the ones there are.
        aggregate: `"torch"` averages in float32 on `device`; `"numpy"` averages in float64.
    """

    def __init__(self, scores_map, generations=5, device=None, with_replacement=False,
                 aggregate="torch"):
        if aggregate not in ("torch", "numpy"):
            raise ValueError(f"aggregate must be 'torch' or 'numpy', got {aggregate!r}")
        self.scores_map = scores_map
        self.generations = generations
        self.device = device
        self.with_replacement = with_replacement
        self.aggregate = aggregate

    def _mean(self, values):
        if self.aggregate == "numpy":
            return float(np.mean(values))
        return float(torch.tensor(values, device=self.device).float().mean())

    def rewards(self, prompt, models):
        out = {}
        for m in models:
            scores = self.scores_map[prompt][m]
            if not isinstance(scores, (list, tuple)):
                out[m] = float(scores)
                continue
            if not scores:
                out[m] = 0.0
                continue
            k = self.generations
            if self.with_replacement and k > len(scores):
                drawn = random.choices(scores, k=k)
            else:
                drawn = random.sample(scores, min(len(scores), k))
            out[m] = self._mean(drawn)
        return out
