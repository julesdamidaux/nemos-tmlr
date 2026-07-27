import os
import json
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from nemos.algorithms.nemos import NeMoS
from nemos.rewards import METRICS, OnlineRewards, PrecomputedRewards, load_score_map
plt.rcParams['text.usetex'] = False

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
BUDGET    = int(0.25*T)
num_runs  = 10
generations  = 5
max_prompts  = 100000
theta_list = [0.1, 0.5, 1, 2, 4]
epsilon_dict = {0.01: lambda t: 0.32, 0.1: lambda t: 0.30, 0.5: lambda t: 0.26, 1: lambda t: 0.26, 2: lambda t: 0.26, 3: lambda t: 0.32, 4: lambda t: 0.35}

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

def single_run(prompts, reward_source, embeddings, X):
    N = len(prompts)
    bandits = {theta: NeMoS(selected_models, X, theta=theta) for theta in theta_list}
    budgets = {theta: BUDGET for theta in theta_list}

    metrics_o2b = {theta: [] for theta in theta_list}
    metrics_opr = {theta: [] for theta in theta_list}
    budgets_used = {theta: [] for theta in theta_list}

    base_metrics = {"optimal": [], "always": [], "random": []}
    base_opr = {"always": [], "random": []}

    for t in tqdm(range(1, T+1), desc="Test iterations"):
        i = random.randint(0, N-1)
        p = prompts[i]
        emb = embeddings[p]

        smap   = reward_source.rewards(p, selected_models)
        base = smap[baseline_model]
        best_m = max(smap, key=smap.get)
        best_s = smap[best_m]

        base_metrics["optimal"].append(best_s - base)
        base_metrics["always"].append(0.0)
        base_opr["always"].append(int(baseline_model == best_m))

        r = random.choice(selected_models)
        base_metrics["random"].append(smap[r] - base)
        base_opr["random"].append(int(r == best_m))

        for theta in theta_list:
            bandit = bandits[theta]
            if budgets[theta] < 0: continue
            arm, delta_k, ucb = bandit.select_arm(i, t)
            eps_func = epsilon_dict[theta]
            query = delta_k < eps_func(t) and budgets[theta] > 0

            if query:
                for m in selected_models:
                    bandit.update(i, m, smap[m])
                reward_k = smap[arm]
                budgets[theta] -= 1
            else:
                reward_k = smap[arm]
                bandit.update(i, arm, reward_k)

            metrics_o2b[theta].append(reward_k - base)
            metrics_opr[theta].append(int(arm == best_m))
            budgets_used[theta].append(BUDGET - budgets[theta])

    return base_metrics, base_opr, metrics_o2b, metrics_opr, budgets_used

# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)

    all_o2b = defaultdict(list)
    all_opr = defaultdict(list)
    all_budgets = defaultdict(list)
    base_o2b = defaultdict(list)
    base_opr = defaultdict(list)

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        b_metrics, b_opr, o2b, opr, budgets = single_run(prompts, reward_source, embeddings, X)
        for k in b_metrics: base_o2b[k].append(b_metrics[k])
        for k in b_opr: base_opr[k].append(b_opr[k])
        for theta in theta_list:
            all_o2b[theta].append(o2b[theta])
            all_opr[theta].append(opr[theta])
            all_budgets[theta].append(budgets[theta])

    avg_o2b = {k: np.mean(np.stack(all_o2b[k]), axis=0) for k in all_o2b}
    avg_opr = {k: np.mean(np.stack(all_opr[k]), axis=0) for k in all_opr}
    avg_bud = {k: np.mean(np.stack(all_budgets[k]), axis=0) for k in all_budgets}
    base_o2b = {k: np.mean(np.stack(base_o2b[k]), axis=0) for k in base_o2b}
    base_opr = {k: np.mean(np.stack(base_opr[k]), axis=0) for k in base_opr}

    styles = {
        "optimal": {"linestyle": "-.", "color": "green"},
        "always": {"linestyle": "--", "color": "blue"},
        "random": {"linestyle": ":", "color": "orange"},
    }
    for i, theta in enumerate(theta_list):
        styles[f"theta_{theta}"] = {"linestyle": "-", "color": plt.cm.viridis(i / len(theta_list))}

    fig, axes = plt.subplots(1,4,figsize=(24,6))

    # 1) Cumulative Regret
    ax = axes[0]
    for theta in theta_list:
        r = np.cumsum(base_o2b["optimal"] - avg_o2b[theta])
        ax.plot(np.arange(1, len(r)+1), r, label=f"$\\theta=${theta}", **styles[f"theta_{theta}"])
    ax.set_title("Cumulative Regret"); ax.set_xlabel("Itération"); ax.set_ylabel("Regret")
    ax.legend(); ax.grid(True)

    # 2) Sliding-window Avg OPR
    ax = axes[1]
    window = T//10
    for theta in theta_list:
        v = avg_opr[theta]
        if len(v) >= window:
            mov = np.convolve(v, np.ones(window)/window, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=f"$\\theta=${theta}", **styles[f"theta_{theta}"])
    ax.set_title(f"{window}-Sliding Avg OPR"); ax.set_xlabel("Itération"); ax.set_ylabel("Avg OPR")
    ax.legend(); ax.grid(True)

    # 3) Budget Consumption
    ax = axes[2]
    for theta in theta_list:
        ax.plot(avg_bud[theta], label=f"$\\theta=${theta}", **styles[f"theta_{theta}"])
    ax.set_title("Budget Consumption"); ax.set_xlabel("Itération"); ax.set_ylabel("Requêtes GT")
    ax.legend(); ax.grid(True)

    # 4) Sliding-window Avg O2B
    ax = axes[3]
    window = T//10
    for theta in theta_list:
        v = avg_o2b[theta]
        if len(v) >= window:
            mov = np.convolve(v, np.ones(window)/window, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=f"$\\theta=${theta}", **styles[f"theta_{theta}"])
    ax.set_title(f"{window}-Sliding Avg O2B"); ax.set_xlabel("Itération"); ax.set_ylabel("Avg O2B")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/theta", exist_ok=True)
    plt.savefig(f"results/theta/results_knn_ucb_theta_comparison_{dataset}_{T}_{num_runs}.pdf")
    plt.close(fig)
