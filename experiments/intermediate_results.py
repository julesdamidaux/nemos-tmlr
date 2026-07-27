import os
import json
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from nemos.algorithms.nemos import NeMoS, phi
from nemos.algorithms._ucb_math import arm_ucb_index
from nemos.rewards import METRICS, OnlineRewards, PrecomputedRewards, load_score_map

# -------- Configuration --------
dataset        = "carrot-bowl"
metric         = "clip"          # "clip" | "imagereward" | "hpsv2"
reward_mode    = "precomputed"   # "precomputed": replay datasets/; "online": generate images
distance = "cosine"  # "cosine" or "L2"
metadata_path  = f"../datasets/{dataset}/{METRICS[metric][0]}"
prompts_path   = f"../datasets/prompts/{dataset}.json"
selected_models= ["Sana", "Unidiffuser", "LCM", "Koala", "SDXL-Turbo", "SSD-1B"]
baseline_model = "SSD-1B"
assert baseline_model in selected_models

T         = 2000
BUDGET20  = int(0.2*T)

epsilon_balrog = 0.22
THETA        = 1.0
num_runs     = 1
generations  = 5
max_prompts  = 10000

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

def get_reward_estimates(bandit, x_idx, t):
    """NeMoS per-model reward estimate f_hat_g(X_t, k_g) at round t.

    Uses the same per-arm construction as the algorithm (nemos.py) so the
    estimation-error analysis tracks the real estimator: f_hat = I_g - U_g.
    """
    if not bandit.history:
        return {m: 0.0 for m in bandit.models}

    x = bandit.emb[x_idx : x_idx + 1]
    phi_t = phi(t)

    # Group each model's own past observations.
    hist_by_model = {m: [] for m in bandit.models}
    for (pidx, m, r) in bandit.history:
        if m in hist_by_model:
            hist_by_model[m].append((pidx, r))

    estimates = {}
    for model in bandit.models:
        entries = hist_by_model[model]
        n_g = len(entries)
        if n_g == 0:
            estimates[model] = 0.0
            continue

        idxs_m = [e[0] for e in entries]
        rewards_m = [e[1] for e in entries]
        hist_x = bandit.emb[idxs_m]

        if bandit.distance == "L2":
            dists = torch.norm(x - hist_x, p=2, dim=1)
        else:
            cos_sim = torch.nn.functional.cosine_similarity(x, hist_x, dim=1)
            dists = 1.0 - cos_sim

        order = torch.argsort(dists)
        rewards_sorted = [rewards_m[j] for j in order.tolist()]
        dists_sorted = dists[order].tolist()

        ucb, bonus, _, _ = arm_ucb_index(
            rewards_sorted, dists_sorted, n_g, bandit.theta, phi_t, bandit.beta
        )
        estimates[model] = ucb - bonus  # f_hat at k* = I_g - U_g

    return estimates

def single_run(prompts, reward_source, embeddings, X, algo_name):
    N = len(prompts)
    estimation_errors = {m: [] for m in selected_models}  # Store errors per model over time
    
    if algo_name == "NeMoS 20":
        bandit = NeMoS(selected_models, X, theta=THETA, distance=distance)
        budget_remaining = BUDGET20
    else:  # KNN-UCB
        bandit = NeMoS(selected_models, X, theta=1.0, distance=distance)
    
    if T <= N:
        idxs = random.sample(range(N), T)
    else:
        idxs = random.choices(range(N), k=T)

    for t in tqdm(range(1, T+1), desc=f"{algo_name}"):
        i = idxs[t-1]
        p = prompts[i]
        emb = embeddings[p]

        smap   = reward_source.rewards(p, selected_models)

        # Get estimated rewards for all models
        estimates = get_reward_estimates(bandit, i, t)
        
        # Calculate absolute estimation errors for all models
        for model in selected_models:
            true_reward = smap[model]
            estimated_reward = estimates[model]
            error = abs(estimated_reward - true_reward)
            estimation_errors[model].append(error)
        
        # Select arm
        arm, delta_k, _, _, _ = bandit.select_arm(i, t)
        
        if algo_name == "NeMoS 20":
            # NeMoS logic with active learning
            if delta_k < epsilon_balrog and budget_remaining > 0:
                for m in selected_models:
                    bandit.update(i, m, smap[m])
                budget_remaining -= 1
            else:
                bandit.update(i, arm, smap[arm])
        else:
            # KNN-UCB: passive only
            bandit.update(i, arm, smap[arm])

    return estimation_errors


# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)
    
    algos = ["NeMoS 20", "KNN-UCB"]
    all_errors = {algo: {m: [] for m in selected_models} for algo in algos}
    
    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Running {algo}")
        print(f"{'='*60}")
        
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            errors = single_run(prompts, reward_source, embeddings, X, algo)
            for model in selected_models:
                all_errors[algo][model].append(errors[model])  # Keep as list per run
    
    # Save results
    os.makedirs("results/intermediate_results/data", exist_ok=True)
    save_path = os.path.join(
        "results",
        "intermediate_results",
        "data",
        f"estimation_errors_{dataset}_{T}_{num_runs}runs.pkl"
    )
    with open(save_path, "wb") as f:
        pickle.dump(all_errors, f)
    print(f"\nSaved estimation errors to {save_path}")
    
    # Compute average errors over runs for each model
    avg_errors = {}
    for algo in algos:
        avg_errors[algo] = {}
        for model in selected_models:
            avg_errors[algo][model] = np.mean(np.stack(all_errors[algo][model]), axis=0)
    
    # Compute average across all models at each iteration
    avg_across_models = {}
    for algo in algos:
        # Stack all model errors
        all_model_errors = np.stack([avg_errors[algo][m] for m in selected_models], axis=0)
        # Average across models at each iteration
        avg_across_models[algo] = np.mean(all_model_errors, axis=0)
    
    # Apply sliding window average
    window = 500
    sliding_avg_errors = {}
    for algo in algos:
        errors = avg_across_models[algo]
        if len(errors) >= window:
            sliding_avg = np.convolve(errors, np.ones(window)/window, mode="valid")
            sliding_avg_errors[algo] = sliding_avg
        else:
            sliding_avg_errors[algo] = errors
    
    # Plotting
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = {"NeMoS 20": "indigo", "KNN-UCB": "magenta"}
    linestyles = {"NeMoS 20": "-", "KNN-UCB": "--"}
    
    for algo in algos:
        errors = sliding_avg_errors[algo]
        start_iter = window if len(avg_across_models[algo]) >= window else 1
        ax.plot(np.arange(start_iter, start_iter + len(errors)), errors, 
               color=colors[algo], linestyle=linestyles[algo], 
               linewidth=2.5, label=algo)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'{window}-Sliding Avg Absolute Error (Avg over Models)')
    ax.set_title('Estimation Error Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    os.makedirs("results/intermediate_results", exist_ok=True)
    plt.savefig(f"results/intermediate_results/estimation_errors_{dataset}_{T}_{num_runs}runs.pdf", dpi=600, bbox_inches="tight")
    print(f"\nSaved plot to results/intermediate_results/estimation_errors_{dataset}_{T}_{num_runs}runs.pdf")

