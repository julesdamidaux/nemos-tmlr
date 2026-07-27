# n_models_queries.py
import os, json, pickle, random, math
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

from nemos.algorithms.nemos import NeMoS  # doit inclure .rank_arms()
from nemos.rewards import METRICS, OnlineRewards, PrecomputedRewards, load_score_map

# -------- Config --------
dataset        = "ms-coco"
metric         = "clip"          # "clip" | "imagereward" | "hpsv2"
reward_mode    = "precomputed"   # "precomputed": replay datasets/; "online": generate images
metadata_path  = f"../datasets/{dataset}/{METRICS[metric][0]}"
prompts_path   = f"../datasets/prompts/{dataset}.json"

selected_models= ["Sana", "Unidiffuser", "LCM", "Koala", "SDXL-Turbo", "SSD-1B"]
baseline_model = "SDXL-Turbo"
assert baseline_model in selected_models

T            = 5000
generations  = 5
num_runs     = 5
max_prompts  = 3000
THETA        = 1.0

# Test these K in parallel; "full" = all models (thus K=6 here)
K_grid = [2, 3, 4, 5, "full"]

# -------- Helpers --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_prompt_embedding(prompt, cache={}):
    if prompt not in cache:
        inp = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inp)
        cache[prompt] = (feat / feat.norm(dim=-1, keepdim=True)).squeeze(0)
    return cache[prompt]

def load_data(path, max_load):
    """Prompt set and its embeddings; the offline score map too, when replaying."""
    if reward_mode == "online":
        with open(prompts_path, "r", encoding="utf-8") as f:
            valid = json.load(f)
        scores_map = None
    else:
        scores_map, valid = load_score_map(path, selected_models, metric)
    prompts = random.sample(valid, min(max_load, len(valid)))
    if len(prompts) == 0:
        raise ValueError("No valid prompt found — check your metadata / filters.")
    embeddings = {p: get_prompt_embedding(p) for p in tqdm(prompts, desc="Embedding prompts")}
    X = torch.stack([embeddings[p] for p in prompts], dim=0)
    return prompts, scores_map, embeddings, X

def build_reward_source(scores_map):
    """Replay the offline scores, or generate and score an image at every step."""
    if reward_mode == "online":
        return OnlineRewards(metric=metric, generations=generations, device=device,
                             cache_path=f"../datasets/{dataset}/online_scores_{metric}.json")
    return PrecomputedRewards(scores_map, generations=generations, device=device, with_replacement=True,
                              aggregate="numpy")

def epsilon_from_budget(budget_rate):
    """Legacy calibration: eps(5%)=0.065, eps(25%)=0.25; linear interpolation between the two."""
    b0, e0 = 0.05, 0.065
    b1, e1 = 0.25, 0.25
    if budget_rate <= b0: return e0
    if budget_rate >= b1: return e1
    w = (budget_rate - b0) / (b1 - b0)
    return e0*(1-w) + e1*w


def budget_rate_for_K(K, total_models):
    """Compute-equivalent (25%) : budget_rate(K) = 0.25 / (K-1)."""
    K_eff = total_models if K == "full" else int(K)
    if K_eff <= 1:
        raise ValueError("K must be >=2 for compute-equivalent mapping (division by zero).")
    return 0.25 / (K_eff - 1)

def make_indices(N, T):
    """Select T prompt indices.
       - if T <= N: sampling without replacement
       - if T > N: sampling with replacement (fixes random.sample error)
    """
    if N <= 0:
        raise ValueError("Number of prompts N=0 — impossible to construct evaluation order.")
    return random.sample(range(N), T) if T <= N else random.choices(range(N), k=T)

# ========= MAIN =========
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)

    total_models = len(selected_models)
    # Prepare configs (K, budget_rate(K)) at compute-equivalent 25%
    configs = []
    for K in K_grid:
        K_eff = total_models if K == "full" else int(K)
        b_rate = budget_rate_for_K(K, total_models)
        label  = f"NeMoS (K={K_eff}, budget={100*b_rate:.2f}%)"
        configs.append({"K": K, "K_eff": K_eff, "budget_rate": b_rate, "label": label})

    # Stockage global
    all_results = { c["K"]: {"o2b_runs": [], "bud_runs": []} for c in configs }
    optimal_runs = []

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")

        N = len(prompts)
        idxs = make_indices(N, T)  # <-- FIX: supporte T > N (avec remplacement)

        # init bandit + budgets par config
        state = {}
        for c in configs:
            eps = epsilon_from_budget(c["budget_rate"])
            state[c["K"]] = {
                "nb": NeMoS(selected_models, X, theta=THETA),
                "budget": int(c["budget_rate"] * T),
                "eps": eps,
                "o2b": [],
                "bud": []
            }

        o2b_opt = []

        for t in tqdm(range(1, T+1), desc="Synced iterations"):
            i = idxs[t-1]
            p = prompts[i]

            # same instantaneous estimates for ALL configs
            smap = reward_source.rewards(p, selected_models)
            base   = smap[baseline_model]
            best_m = max(smap, key=smap.get)
            best_s = smap[best_m]
            o2b_opt.append(best_s - base)

            for c in configs:
                st = state[c["K"]]
                nb, budget, eps = st["nb"], st["budget"], st["eps"]

                arm, delta_k, _, _, _ = nb.select_arm(i, t)

                if delta_k < eps and budget > 0:
                    ranking = nb.rank_arms(i, t)  # [(model, ucb, bonus, var), ...]
                    if c["K"] == "full":
                        to_query = [m for (m, _, _, _) in ranking]
                    else:
                        to_query = [ranking[j][0] for j in range(min(c["K_eff"], len(ranking)))]
                    for m in to_query:
                        nb.update(i, m, smap[m])
                    reward = smap[arm]
                    budget -= 1
                else:
                    reward = smap[arm]
                    nb.update(i, arm, reward)

                st["o2b"].append(reward - base)
                st["bud"].append(int((int(c["budget_rate"]*T) - budget)))
                st["budget"] = budget

        # stack run
        for c in configs:
            all_results[c["K"]]["o2b_runs"].append(np.asarray(state[c["K"]]["o2b"], dtype=float))
            all_results[c["K"]]["bud_runs"].append(np.asarray(state[c["K"]]["bud"], dtype=float))
        optimal_runs.append(np.asarray(o2b_opt, dtype=float))

    # Save
    os.makedirs("results/n_models_queries/data", exist_ok=True)
    out_path = f"results/n_models_queries/data/n_models_queries_compute_equiv_{dataset}_{T}_{num_runs}runs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "configs": configs,
            "all_results": all_results,
            "optimal_runs": optimal_runs,
            "selected_models": selected_models,
            "baseline_model": baseline_model,
            "generations": generations,
            "T": T
        }, f)
    print(f"[OK] Saved {out_path}")

    # =========================
    # PLOTTING (1x2): OtB + Budget
    # =========================
    w = T // 10
    def sliding_avg(x, w):
        if len(x) < w: return np.array([])
        return np.convolve(x, np.ones(w)/w, mode="valid")

    # styles par K
    color_map = {2:"tab:orange", 3:"tab:green", 4:"tab:red", 5:"tab:purple", "full":"tab:blue"}
    linew = 2.4

    # Compute averages over runs
    avg_o2b = {}
    avg_bud = {}
    label_map = {c["K"]: c["label"] for c in configs}

    for c in configs:
        lab = label_map[c["K"]]
        o2b_runs = all_results[c["K"]]["o2b_runs"]
        bud_runs = all_results[c["K"]]["bud_runs"]
        avg_o2b[lab] = np.mean(np.stack(o2b_runs), axis=0)
        avg_bud[lab] = np.mean(np.stack(bud_runs), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    idx = np.linspace(0, T - w, 120, dtype=int)

    # (1) Sliding Avg OtB
    ax = axes[0]
    for c in configs:
        lab = label_map[c["K"]]
        series = avg_o2b[lab]
        if len(series) >= w:
            mov = sliding_avg(series, w)
            ax.plot(np.arange(w, w+len(mov))[idx], mov[idx],
                    label=lab, color=color_map.get(c["K"], "black"), linewidth=linew)
    ax.set_title(f"{w}-Sliding Avg OtB — compute-equivalent (25%) — {dataset}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Avg OtB")
    ax.grid(True); ax.legend()

    # (2) Budget Consumption
    ax = axes[1]
    for c in configs:
        lab = label_map[c["K"]]
        b = avg_bud[lab]
        ax.plot(np.arange(1, len(b)+1), b,
                label=lab, color=color_map.get(c["K"], "black"), linewidth=linew)
    ax.set_title("Budget Consumption (avg over runs)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("GT Queries (cumulative)")
    ax.grid(True); ax.legend()

    plt.tight_layout()
    os.makedirs("results/n_models_queries", exist_ok=True)
    plt.savefig(f"results/n_models_queries/curves_compute_equiv_{dataset}_{T}_{num_runs}runs_2panel.pdf", dpi=300)
    plt.show()
