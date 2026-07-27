"""Rewards computed on the fly: generate an image at every step, then score it.

This is the *online* protocol — no metadata file, no replay. Each call generates
`generations` images with the requested model and returns their mean metric score.

Cost, before choosing this over :class:`~nemos.rewards.precomputed.PrecomputedRewards`:
a round evaluates every candidate model, so it costs `len(models) * generations` image
generations (30 with the paper's six models and 5 generations). A 2000-round experiment
therefore means tens of thousands of diffusion samples. Scores are cached to disk per
(model, prompt) so repeated prompts — and reruns — are not regenerated.

Because images are sampled afresh, this protocol does not reproduce the paper's tables;
it measures the same quantity under a different, unseeded sampling process.
"""
import json
import os

import torch

from nemos.rewards.base import RewardSource
from nemos.rewards.pipelines import build_generator, is_black_image

CLIP_ID = "openai/clip-vit-base-patch32"


class OnlineRewards(RewardSource):
    """Generate with each model at every step and score the images immediately.

    Args:
        metric: `"clip"`, `"imagereward"` or `"hpsv2"`.
        generations: images generated per (prompt, model).
        device: torch device; defaults to CUDA when available.
        cache_path: JSON file accumulating `{model: {prompt: [scores]}}`. Reused across
            runs; pass None to disable caching.
        hps_version: HPSv2 checkpoint variant, when `metric="hpsv2"`.
        keep_loaded: keep every pipeline resident (fast, needs VRAM for all models at
            once). False frees each pipeline after use — much slower, much less VRAM.
        skip_black: drop all-black generations, as the dataset builder does. If every
            image of a (prompt, model) pair is black, the unfiltered images are scored
            rather than leaving the bandit without a reward.
    """

    def __init__(self, metric="clip", generations=5, device=None, cache_path=None,
                 hps_version="v2.1", keep_loaded=True, skip_black=True):
        self.metric = metric
        self.generations = generations
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_path = cache_path
        self.hps_version = hps_version
        self.keep_loaded = keep_loaded
        self.skip_black = skip_black

        self._generators = {}
        self._scorer = None
        self._cache = self._load_cache()

    # ----- cache -----

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    # ----- scoring -----

    def _get_scorer(self):
        """Return `scorer(prompt, images) -> list[float]` for the configured metric."""
        if self._scorer is not None:
            return self._scorer

        if self.metric == "clip":
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
            processor = CLIPProcessor.from_pretrained(CLIP_ID)

            def score(prompt, images):
                out = []
                for image in images:
                    inputs = processor(text=[prompt], images=image, return_tensors="pt",
                                      padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    img = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    txt = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    out.append(100 * (img * txt).sum(dim=-1).item())
                return out

        elif self.metric == "imagereward":
            from ImageReward import load

            model = load("ImageReward-v1.0")
            model.device = self.device

            def score(prompt, images):
                return [float(s) for s in model.score(prompt, images)]

        elif self.metric == "hpsv2":
            import hpsv2

            def score(prompt, images):
                return [float(s) for s in hpsv2.score(images, prompt,
                                                      hps_version=self.hps_version)]

        else:
            raise ValueError(f"Unknown metric {self.metric!r}")

        self._scorer = score
        return score

    # ----- generation -----

    def _generate(self, prompt, model):
        generator = self._generators.get(model)
        if generator is None:
            generator = build_generator(model, self.device)
            self._generators[model] = generator

        images = [generator(prompt) for _ in range(self.generations)]

        if not self.keep_loaded:
            del self._generators[model]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.skip_black:
            kept = [img for img in images if not is_black_image(img)]
            if kept:
                return kept
            print(f"Warning: every generation of {model} on {prompt!r} was black; "
                  f"scoring them unfiltered")
        return images

    # ----- RewardSource -----

    def scores(self, prompt, model):
        """Per-image scores of `model` on `prompt`, from cache when available."""
        cached = self._cache.get(model, {}).get(prompt)
        if cached:
            return cached

        images = self._generate(prompt, model)
        scores = self._get_scorer()(prompt, images)
        self._cache.setdefault(model, {})[prompt] = scores
        self._save_cache()
        return scores

    def rewards(self, prompt, models):
        out = {}
        for m in models:
            scores = self.scores(prompt, m)
            out[m] = float(torch.tensor(scores, device=self.device).float().mean())
        return out
