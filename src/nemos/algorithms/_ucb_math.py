"""Pure-Python core of the NeMoS UCB index (no torch), kept separate so the
algorithm's math can be unit-tested independently of the tensor plumbing.

This implements the construction of Section 4 of the paper:
    f_hat_g(X_t, k) = (1/k) * sum of the k nearest rewards                 (Eq. 2)
    U_g(X_t, k)     = sqrt(theta * log N_g(t) / k) + phi(t) * r_{g,k}(t)   (Eq. 3)
    k_g(t)          = argmin_{1<=k<=N_g(t)} U_g(X_t, k)                    (Eq. 4)
    I_g(X_t)        = f_hat_g(X_t, k_g) + U_g(X_t, k_g)                    (Eq. 5)
"""
import math

import numpy as np


def arm_ucb_index(rewards_by_dist, dists_by_dist, n_g, theta, phi_t, beta=1.0):
    """UCB index for a single model.

    Args:
        rewards_by_dist: this model's past rewards, ordered by increasing
            distance of the corresponding prompt to the current prompt X_t.
        dists_by_dist: the matching distances (r_{g,k} = dists_by_dist[k-1],
            i.e. the distance to the k-th nearest neighbour).
        n_g: N_g(t), the number of past observations of this model
            (== len(rewards_by_dist)).
        theta: UCB exploration constant (theta = 1 in the paper).
        phi_t: phi(t) already evaluated (paper: phi(t) = log t).
        beta: weight on the geometric term (1 in the paper).

    Returns:
        (I_g, U_g at k*, variance of the k* nearest rewards, k*).
        For a model with no history returns (+inf, +inf, 0, 0) so the caller
        forces initial exploration.
    """
    if n_g <= 0:
        return math.inf, math.inf, 0.0, 0

    # N_g(t) >= 1 here, so log is well defined (log 1 = 0, no log(0)/division issue).
    log_ng = math.log(n_g)

    best_ucb, best_bonus, best_k = -math.inf, math.inf, 0
    best_scores = []
    running_sum = 0.0
    for k in range(1, n_g + 1):
        running_sum += rewards_by_dist[k - 1]
        mu_hat = running_sum / k
        bonus = math.sqrt(theta * log_ng / k) + phi_t * beta * dists_by_dist[k - 1]
        ucb = mu_hat + bonus
        if bonus < best_bonus:  # k_g(t) = argmin_k U_g(X_t, k)
            best_ucb, best_bonus, best_k = ucb, bonus, k
            best_scores = rewards_by_dist[:k]

    var_k = float(np.var(best_scores)) if best_scores else 0.0
    return best_ucb, best_bonus, var_k, best_k
