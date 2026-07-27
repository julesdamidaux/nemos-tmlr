"""The reward interface shared by every experiment.

An experiment never computes rewards itself: it asks a :class:`RewardSource` for the
reward of each candidate model on the current prompt. Two implementations exist:

* :class:`~nemos.rewards.precomputed.PrecomputedRewards` — replays scores computed
  offline (the protocol used for the paper's tables);
* :class:`~nemos.rewards.online.OnlineRewards` — generates an image with each model at
  every step and scores it on the spot.
"""
from abc import ABC, abstractmethod


class RewardSource(ABC):
    """Maps a (prompt, model) pair to a scalar reward."""

    @abstractmethod
    def rewards(self, prompt, models):
        """Reward of every model in `models` on `prompt`.

        Args:
            prompt: the prompt (text-to-image) or question (LLM selection).
            models: model names to evaluate, in the order they should be evaluated.
                Order matters: implementations that draw random samples consume the
                RNG in this order.

        Returns:
            dict mapping each name in `models` to a float reward.
        """

    def reward(self, prompt, model):
        """Reward of a single model — convenience wrapper around :meth:`rewards`."""
        return self.rewards(prompt, [model])[model]
