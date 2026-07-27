import os
import json
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from nemos.algorithms.nemos import NeMoS
from nemos.rewards import METRICS, OnlineRewards, PrecomputedRewards, load_score_map

# -------- Configuration --------
dataset        = "carrot-bowl"
metric         = "clip"          # "clip" | "imagereward" | "hpsv2"
reward_mode    = "precomputed"   # "precomputed": replay datasets/; "online": generate images
metadata_path  = f"../datasets/{dataset}/{METRICS[metric][0]}"
prompts_path   = f"../datasets/prompts/{dataset}.json"
selected_models= ["Sana", "Unidiffuser", "LCM", "SDXL-Turbo", "Koala", 'SSD-1B']
baseline_model = "SSD-1B"
assert baseline_model in selected_models

T         = 2000
BUDGET    = int(0.2*T)
epsilon   = lambda t: 0.23
ucb_limit = 2.8
var_limit = 5.5
THETA     = 1.0
num_runs  = 5
generations  = 5
max_prompts  = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------- CLIP Setup --------
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
    embeddings = {p: get_prompt_embedding(p) for p in tqdm(prompts, desc="Embedding prompts")}
    X = torch.stack([embeddings[p] for p in prompts], dim=0)
    return prompts, scores_map, embeddings, X

def build_reward_source(scores_map):
    """Replay the offline scores, or generate and score an image at every step."""
    if reward_mode == "online":
        return OnlineRewards(metric=metric, generations=generations, device=device,
                             cache_path=f"../datasets/{dataset}/online_scores_{metric}.json")
    return PrecomputedRewards(scores_map, generations=generations, device=device)

def single_run(prompts, reward_source, embeddings, X, algos):
    N = len(prompts)
    metrics_o2b = {alg: [] for alg in algos}
    metrics_opr = {alg: [] for alg in algos if alg != "optimal"}
    budgets     = {}
    for variant in ["Delta", "Warm-start", "UCB", "NN-Var", "Random-p"]:
        if variant in algos:
            budgets[variant] = []

    bandits_knn = {
        "Delta": NeMoS(selected_models, X, theta=THETA),
        "Warm-start": NeMoS(selected_models, X, theta=THETA),
        "UCB": NeMoS(selected_models, X, theta=THETA),
        "No AL": NeMoS(selected_models, X, theta=THETA),
        "NN-Var": NeMoS(selected_models, X, theta=THETA),
        "Random-p": NeMoS(selected_models, X, theta=THETA)
    }
    budgets_knn = {
        "Delta": BUDGET,
        "Warm-start": BUDGET,
        "UCB": BUDGET,
        "NN-Var": BUDGET,
        "Random-p": BUDGET,
    }

    for t in tqdm(range(1, T+1), desc="Test iterations"):
        i = random.randint(0, N-1)
        p = prompts[i]
        emb = embeddings[p]

        smap   = reward_source.rewards(p, selected_models)
        base   = smap[baseline_model]
        best_m = max(smap, key=smap.get)
        best_s = smap[best_m]

        if "optimal" in algos:
            metrics_o2b["optimal"].append(best_s - base)

        if "always" in algos:
            metrics_o2b["always"].append(0.0)
            metrics_opr["always"].append(int(baseline_model == best_m))

        if "random" in algos:
            r = random.choice(selected_models)
            metrics_o2b["random"].append(smap[r] - base)
            metrics_opr["random"].append(int(r == best_m))

        for variant in ["Delta", "Warm-start", "UCB", "No AL", "NN-Var", "Random-p"]:
            if variant not in algos: continue
            bandit = bandits_knn[variant]
            if variant != "No AL":
                budget = budgets_knn[variant]

            arm, delta_k, ucb, var, _ = bandit.select_arm(i, t)

            query = False
            if variant == "Delta":
                query = (delta_k < epsilon(t) and budget > 0)
            elif variant == "Warm-start":
                query = (t < BUDGET and budget > 0)
            elif variant == "UCB":
                query = (ucb > ucb_limit  and budget > 0)
            elif variant == "No AL":
                query = False  # never query
            elif variant == "NN-Var":
                query = (var > var_limit and budget > 0)
            elif variant == "random-p":
                # Query with probability B/T at each timestep
                query_prob = BUDGET / T
                query = (random.random() < query_prob and budget > 0)

            if query:
                for m in selected_models:
                    bandit.update(i, m, smap[m])
                reward_k = smap[arm]
                if variant != "No AL":
                    budgets_knn[variant] -= 1
            else:
                reward_k = smap[arm]
                bandit.update(i, arm, reward_k)

            metrics_o2b[variant].append(reward_k - base)
            metrics_opr[variant].append(int(arm == best_m))
            if variant != "No AL":
                budgets[variant].append(BUDGET - budgets_knn[variant])

    return metrics_o2b, metrics_opr, budgets

# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)

    algos = ["Delta", "Warm-start", "UCB", "No AL", "optimal", "NN-Var", "Random-p"]
    all_o2b       = {a: [] for a in algos}
    all_opr       = {a: [] for a in algos if a != "optimal"}
    budgets_accum = {a: [] for a in algos if a not in ["No AL", "optimal"]}

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        o2b, opr, budgets = single_run(prompts, reward_source, embeddings, X, algos)
        for a in algos:
            all_o2b[a].append(o2b[a])
            if a != "optimal":
                all_opr[a].append(opr[a])
        for a, b in budgets.items():
            budgets_acc[a].append(b)

    # --- Save raw data for later plotting ---
    import os, pickle
    save_path = os.path.join(
        "results",
        "query_trigger_strategies",
        "data",
        f"raw_data_uncertainty_methods_{dataset}_{T}_{num_runs}runs.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "all_o2b":       all_o2b,
            "all_opr":       all_opr,
            "budgets_accum": budgets_accum
        }, f)
    print(f"Saved raw data to {save_path}")

    # --- Compute averages ---
    avg_o2b = {a: np.mean(np.stack(all_o2b[a]), axis=0) for a in algos}
    avg_opr = {a: np.mean(np.stack(all_opr[a]), axis=0) for a in all_opr}
    avg_bud = {a: np.mean(np.stack(budgets_accum[a]), axis=0) for a in budgets_accum}

    # --- Vos styles et plotting inchangés ---
    styles = {
        "optimal":    {"linestyle": "-.", "color": "green"},
        "always":     {"linestyle": "--", "color": "blue"},
        "random":     {"linestyle": ":",  "color": "orange"},
        "Delta":      {"linestyle": "-",  "color": "red"},
        "Warm-start": {"linestyle": "--", "color": "purple"},
        "UCB":        {"linestyle": "-.", "color": "brown"},
        "No AL":      {"linestyle": "-",  "color": "black"},
        "NN-Var":     {"linestyle": "-",  "color": "green"},
        "Random-p":   {"linestyle": ":",  "color": "pink"},
    }

    fig, axes = plt.subplots(1,4,figsize=(24,6))

    # 1) Cumulative Regret
    ax=axes[0]
    for a in algos:
        if a != "optimal":
            r = np.cumsum(avg_o2b["optimal"] - avg_o2b[a])
            ax.plot(np.arange(1, len(r)+1), r, label=a, **styles[a])
    ax.set_title("Cumulative Regret"); ax.set_xlabel("Itération"); ax.set_ylabel("Regret")
    ax.legend(); ax.grid(True)

    # 2) Sliding-window Avg OPR
    ax = axes[1]
    window = T//10
    for a, v in avg_opr.items():
        if len(v) >= window:
            mov = np.convolve(v, np.ones(window)/window, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg OPR"); ax.set_xlabel("Itération"); ax.set_ylabel("Avg OPR")
    ax.legend(); ax.grid(True)

    # 3) Budget Consumption
    ax=axes[2]
    for a, b in avg_bud.items():
        ax.plot(b, label=a, **styles[a])
    ax.set_title("Budget Consumption"); ax.set_xlabel("Itération"); ax.set_ylabel("Requêtes GT")
    ax.legend(); ax.grid(True)

    # 4) Sliding-window Avg O2B
    ax=axes[3]
    for a in algos:
        if len(avg_o2b[a]) >= window and a != "optimal":
            mov = np.convolve(avg_o2b[a], np.ones(window)/window, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg O2B"); ax.set_xlabel("Itération"); ax.set_ylabel("Avg O2B")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/query_trigger_strategies", exist_ok=True)
    plt.savefig(f"results/query_trigger_strategies/results_uncertainty_methods_{dataset}_{T}_{num_runs}.pdf")
    plt.close(fig)
