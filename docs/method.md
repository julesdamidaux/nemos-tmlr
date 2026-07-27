# NeMoS — method & results

NeMoS (**Ne**arest neighbors → **Mo**del **S**election) casts prompt-wise model selection as a
**contextual bandit**. At each round a prompt `X_t` arrives, the algorithm picks one of `G`
candidate models, and observes a reward (e.g. the CLIPScore of the generated image). The goal is to
minimize cumulative regret against the per-prompt oracle.

## The algorithm

At round `t`, each model `g` is scored by a **UCB index** `I_g(X_t) = f̂_g + U_g`:

1. **Nearest-neighbor reward estimate** `f̂_g(X_t, k)` — the average reward of `g` over the `k`
   past prompts closest to `X_t` in embedding space (cosine distance on CLIP embeddings). Similar
   prompts are assumed to yield similar rewards, so feedback generalizes across neighbors.

2. **Uncertainty bonus** `U_g(X_t, k) = sqrt(θ · log N_g(t) / k) + φ(t) · r_{g,k}(t)` — a
   statistical term that shrinks with `k`, plus a geometric term that grows with the distance to
   the `k`-th neighbor. `N_g(t)` is the number of past observations of `g`, `φ(t) = log t`, and
   `θ` (=1 in all experiments) controls exploration.

3. **Adaptive neighborhood** — `k` is chosen per model to *minimize* the bonus, balancing the two
   sources of uncertainty. Models with no history get an infinite index (cold-start exploration).
   NeMoS then plays `arg max_g I_g(X_t)`.

4. **Active *Delta* query rule** — if the gap between the top-two indices is below a threshold `δ`
   (a *near-tie*) and query budget remains, NeMoS spends one query to observe **all** models'
   rewards on `X_t`; otherwise it only observes the played model. Concentrating full-feedback
   queries on ambiguous prompts is what accelerates convergence.

The core UCB computation lives in [`src/nemos/algorithms/nemos.py`](../src/nemos/algorithms/nemos.py),
with the pure-math kernel factored into `_ucb_math.py` (torch-free, so the index computation can be
read and checked independently of the tensor plumbing).

## Results

**Scope of these results.** Everything below is measured on the paper's *text-to-image model
selection* task: four prompt datasets, the six models listed in
[reproduce.md](reproduce.md#model-versions), and rewards given by **automatic metrics** — CLIPScore,
ImageReward and HPSv2. They say nothing about human-judged image quality, and the numbers are
specific to this candidate pool and these prompt distributions.

The reward metric is selected per run by the `metric` flag in each experiment's config block
(`"clip"`, `"imagereward"`, `"hpsv2"`). The table below reports **CLIPScore** rewards, the default
configuration, with rewards replayed from the offline datasets — see
[where rewards come from](reproduce.md#where-rewards-come-from).

Cumulative regret across the four prompt datasets, six text-to-image models, averaged over 10 runs
(**lower is better** — Table 1 of the paper):

| Algorithm | MS-COCO | Flickr | Flowers | Carrot-bowl |
|---|:--:|:--:|:--:|:--:|
| Optimal *(oracle)* | 0.000 | 0.000 | 0.000 | 0.000 |
| Always *(best fixed model)* | 1.032 | 1.161 | 0.884 | 1.232 |
| Random | 1.905 | 1.713 | 2.476 | 2.003 |
| neuronal-s | 2.023 | 1.511 | 1.845 | 1.976 |
| PAK-UCB | 1.714 | 1.546 | 2.243 | 1.800 |
| LinUCB | 2.013 | 1.161 | 1.690 | 1.603 |
| KNN-UCB *(passive NeMoS)* | 1.112 | 1.206 | 1.031 | 1.158 |
| **NeMoS (5%)** | 1.016 | 1.141 | 0.953 | 1.032 |
| **NeMoS (20%)** | **0.930** | **0.989** | **0.767** | **0.894** |

- On these four datasets, NeMoS cuts PAK-UCB's cumulative CLIPScore regret by **40–60%**, and a 20%
  query budget shaves a further **15–25%** off the passive KNN-UCB baseline.
- It achieves a **positive Outscore-to-Best** in the automatic metric, i.e. its per-prompt routing
  yields a higher average metric score than the best single model in the pool, and improves OPR by
  ~10 points over the baselines compared here.
- **Robust to a changing model pool** in these experiments: it folds in newly added models and
  reallocates budget when models are removed mid-stream
  (`experiments/model_addition.py`, `experiments/model_removal.py`).
- **Also applies beyond images**: the same method selects between two LLMs on CommonsenseQA, scored
  by answer accuracy (`experiments/compare_to_baselines_LLM.py`).

## Theory

Under the paper's assumptions, with `θ > 2` and an active-query budget `B(T) = T / log T`, NeMoS
achieves a **poly-logarithmic** cumulative regret bound — a large improvement over the near-linear
regret of the passive KNN-UCB baseline.

See the [paper](https://openreview.net/forum?id=CSjewjplO1) for the full figures, ablations, and
the regret proof.
