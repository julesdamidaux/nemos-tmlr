import os
import json
import pickle
import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from nemos.algorithms.nemos import NeMoS
from nemos.rewards import METRICS, OnlineRewards, PrecomputedRewards, load_score_map

# -------- Configuration --------
dataset        = "carrot-bowl"
metric         = "clip"          # "clip" | "imagereward" | "hpsv2"
reward_mode    = "precomputed"   # "precomputed": replay datasets/; "online": generate images
metadata_path  = f"../datasets/{dataset}/{METRICS[metric][0]}"
prompts_path   = f"../datasets/prompts/{dataset}.json"
selected_models= ["Sana", "Unidiffuser", "LCM", "SDXL-Turbo", "SSD-1B", "Koala"]
baseline_model = "SSD-1B"
assert baseline_model in selected_models

T         = 2000
num_runs  = 20
generations  = 5
max_prompts  = 100000
THETA = 1.0

budgets_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]  # fractions of T
epsilon_dict = {
    0.0: lambda t: 0.,
    0.05: lambda t: 0.05,
    0.1: lambda t: 0.10,
    0.2: lambda t: 0.21,
    0.3: lambda t: 0.30,
    0.4: lambda t: 0.39,
}

# -------- CLIP Setup --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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

def single_run(prompts, reward_source, embeddings, X, budgets_list, epsilon_dict):
    N = len(prompts)

    metrics_o2b = defaultdict(list)
    metrics_opr = defaultdict(list)
    budgets_used = {frac: [] for frac in budgets_list}

    bandits = {frac: NeMoS(selected_models, X, theta=THETA) for frac in budgets_list}
    budgets = {frac: int(frac * T) for frac in budgets_list}

    for t in tqdm(range(1, T+1), desc="Test iterations"):
        i = random.randint(0, N-1)
        p = prompts[i]
        emb = embeddings[p]

        smap   = reward_source.rewards(p, selected_models)
        base   = smap[baseline_model]
        best_m = max(smap, key=smap.get)
        best_s = smap[best_m]

        metrics_o2b["optimal"].append(best_s - base)
        metrics_o2b["always"].append(0.0)
        metrics_opr["always"].append(int(baseline_model == best_m))

        r = random.choice(selected_models)
        metrics_o2b["random"].append(smap[r] - base)
        metrics_opr["random"].append(int(r == best_m))

        for frac in budgets_list:
            bandit = bandits[frac]
            budget = budgets[frac]
            epsilon_fn = epsilon_dict[frac]

            arm, delta_k, ucb, _ = bandit.select_arm(i, t)
            query = (delta_k < epsilon_fn(t) and budget > 0)

            if query:
                for m in selected_models:
                    bandit.update(i, m, smap[m])
                reward_k = smap[arm]
                budgets[frac] -= 1
            else:
                reward_k = smap[arm]
                bandit.update(i, arm, reward_k)

            label = f"Budget: {int(frac*100)}%"
            metrics_o2b[label].append(reward_k - base)
            metrics_opr[label].append(int(arm == best_m))
            budgets_used[frac].append(int(frac * T) - budgets[frac])

    return metrics_o2b, metrics_opr, budgets_used

# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)

    all_o2b = defaultdict(list)
    all_opr = defaultdict(list)
    all_budgets = {f"Budget: {int(frac*100)}%": [] for frac in budgets_list}

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        o2b, opr, bud = single_run(prompts, reward_source, embeddings, X, budgets_list, epsilon_dict)
        for a in o2b: all_o2b[a].append(o2b[a])
        for a in opr: all_opr[a].append(opr[a])
        for i, frac in enumerate(budgets_list):
            label = f"Budget: {int(frac*100)}%"
            all_budgets[label].append(bud[frac])

    avg_o2b = {a: np.mean(np.stack(all_o2b[a]), axis=0) for a in all_o2b}
    avg_opr = {a: np.mean(np.stack(all_opr[a]), axis=0) for a in all_opr}
    avg_bud = {a: np.mean(np.stack(all_budgets[a]), axis=0) for a in all_budgets}

    save_path = os.path.join(
        "results",
        "different_budgets",
        "data",
        f"raw_data_budget_comparison_{dataset}_{T}_{num_runs}runs.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "all_o2b":    all_o2b,
            "all_opr":    all_opr,
            "all_budgets": all_budgets
        }, f)
    print(f"Saved raw data to {save_path}")

    styles = {
        "optimal": {"linestyle": "-.", "color": "green"},
        "always": {"linestyle": "--", "color": "blue"},
        "random": {"linestyle": ":", "color": "orange"},
    }
    for idx, frac in enumerate(budgets_list):
        label = f"Budget: {int(frac*100)}%"
        styles[label] = {"linestyle": "-", "color": plt.cm.viridis(idx / len(budgets_list))}

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1) Cumulative Regret
    ax = axes[0]
    for a in avg_o2b:
        if a != "optimal":
            r = np.cumsum(avg_o2b["optimal"] - avg_o2b[a])
            ax.plot(np.arange(1, len(r)+1), r, label=a, **styles[a])
    ax.set_title("Cumulative Regret"); ax.set_xlabel("Itération"); ax.set_ylabel("Regret")
    ax.legend(); ax.grid(True)

    # 2) Sliding-window Avg OPR
    ax = axes[1]
    window = T//10
    for a, v in avg_opr.items():
        if len(v) >= window and a not in ["optimal", "random", 'always']:
            mov = np.convolve(v, np.ones(window)/window, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg OPR"); ax.set_xlabel("Itération"); ax.set_ylabel("Avg OPR")
    ax.legend(); ax.grid(True)

    # 3) Budget Consumption
    ax = axes[2]
    for a, b in avg_bud.items():
        ax.plot(b, label=a, **styles[a])
    ax.set_title("Budget Consumption"); ax.set_xlabel("Itération"); ax.set_ylabel("Requêtes GT")
    ax.legend(); ax.grid(True)

    # 4) Sliding-window Avg O2B
    ax = axes[3]
    window = T//10
    for a in avg_o2b:
        if len(avg_o2b[a]) >= window and a not in ["optimal", "random", "always"]:
            mov = np.convolve(avg_o2b[a], np.ones(window)/window, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg O2B"); ax.set_xlabel("Itération"); ax.set_ylabel("Avg O2B")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/different_budgets", exist_ok=True)
    plt.savefig(f"results/different_budgets/results_knn_ucb_budget_comparison_{dataset}_{T}_{num_runs}.pdf")
    plt.close(fig)