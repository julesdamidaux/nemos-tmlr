"""Where an experiment's rewards come from.

Every experiment talks to a :class:`RewardSource` instead of computing rewards itself:

* :class:`PrecomputedRewards` replays scores from `datasets/<dataset>/metadata*.json` —
  the protocol behind the paper's results, and the default everywhere;
* :class:`OnlineRewards` generates an image with each model at every step and scores it
  on the spot, with no dataset file involved.

Both are constructed by the `build_reward_source` helper at the top of each experiment.
"""
from nemos.rewards.base import RewardSource
from nemos.rewards.online import OnlineRewards
from nemos.rewards.precomputed import (
    METRICS,
    PrecomputedRewards,
    load_score_map,
    metadata_path,
)

__all__ = [
    "RewardSource",
    "PrecomputedRewards",
    "OnlineRewards",
    "METRICS",
    "load_score_map",
    "metadata_path",
]
