import math

import torch

from nemos.algorithms._ucb_math import arm_ucb_index

# -------- Configuration --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# phi(t) = log(t) as in the paper; max(t, 1) guards against log(0) for t <= 0.
phi = lambda t: math.log(max(t, 1))


# -------- KNN-UCB Bandit (NeMoS core; cosine distance by default) --------
class NeMoS:
    def __init__(self, models, embeddings, theta=1.0, beta=1, distance="cosine"):
        """
        models: list of model names
        embeddings: torch.Tensor of shape (num_prompts, embedding_dim)
        distance: "cosine" or "L2"
        """
        self.models = models
        self.emb = embeddings
        self.history = []  # list of tuples (prompt_idx, model_name, reward)
        self.theta = theta
        self.beta = beta
        self.distance = distance

    def _ucb_per_model(self, x_idx, t):
        """
        Computes (UCB index, uncertainty bonus, neighbour variance) for each
        model at round t, following Section 4 of the paper. Neighbours are taken
        within each model's own history H_g(t) (per-arm k-NN). Returns a list
        [(model, ucb, bonus, var)], unsorted.
        """
        if not self.history:
            # No history: every model gets an infinite index to force exploration.
            return [(m, float('inf'), float('inf'), float('inf')) for m in self.models]

        x = self.emb[x_idx : x_idx + 1]
        phi_t = phi(t)

        # Group each model's own past observations: model -> (prompt_idx, reward).
        hist_by_model = {m: [] for m in self.models}
        for (pidx, m, r) in self.history:
            if m in hist_by_model:
                hist_by_model[m].append((pidx, r))

        stats = []
        for m in self.models:
            entries = hist_by_model[m]
            n_g = len(entries)  # N_g(t)
            if n_g == 0:
                # Never played: infinite index (cold start).
                stats.append((m, float('inf'), float('inf'), 0.0))
                continue

            idxs_m = [e[0] for e in entries]
            rewards_m = [e[1] for e in entries]
            hist_x = self.emb[idxs_m]

            # Distance from X_t to each of this model's past prompts.
            if self.distance == "L2":
                dists = torch.norm(x - hist_x, p=2, dim=1)
            else:  # cosine (default)
                cos_sim = torch.nn.functional.cosine_similarity(x, hist_x, dim=1)
                dists = 1.0 - cos_sim

            # Sort this model's history by increasing distance to X_t.
            order = torch.argsort(dists)
            rewards_sorted = [rewards_m[j] for j in order.tolist()]
            dists_sorted = dists[order].tolist()

            ucb, bonus, var_k, _ = arm_ucb_index(
                rewards_sorted, dists_sorted, n_g, self.theta, phi_t, self.beta
            )
            stats.append((m, ucb, bonus, var_k))

        return stats

    def rank_arms(self, x_idx, t):
        """Models sorted by decreasing UCB index: [(model, ucb, bonus, var), ...]."""
        stats = self._ucb_per_model(x_idx, t)
        stats.sort(key=lambda z: z[1], reverse=True)
        return stats

    def select_arm(self, x_idx, t):
        """
        Returns: best_model, delta_ucb, best_bonus, best_var, second_model
        where delta_ucb is the gap between the top two UCB indices (used by the
        Delta active-query rule).
        """
        stats = self.rank_arms(x_idx, t)
        best_model, best_ucb, best_bonus, best_var = stats[0]
        if len(stats) > 1:
            second_model, second_ucb, _, _ = stats[1]
        else:
            second_model, second_ucb = None, -float('inf')
        delta_ucb = best_ucb - second_ucb
        return best_model, delta_ucb, best_bonus, best_var, second_model

    def update(self, x_idx, arm, reward):
        """Append a new observation (prompt_idx, model_name, reward)."""
        self.history.append((x_idx, arm, reward))
