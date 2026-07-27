import os
import json
import pickle
import random
import torch
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
distance = "cosine"  # "cosine" or "L2"
metadata_path  = f"../datasets/{dataset}/{METRICS[metric][0]}"
prompts_path   = f"../datasets/prompts/{dataset}.json"
selected_models= ["Sana", "Unidiffuser", "LCM", "Koala", "SDXL-Turbo", "SSD-1B"]
baseline_model = "SSD-1B"
assert baseline_model in selected_models

T         = 2000
BUDGET    = int(0.2*T)

# Different delta values to test
delta_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

THETA        = 1.0
num_runs     = 20
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

def single_run(prompts, reward_source, embeddings, X, delta):
    N = len(prompts)
    metrics_o2b = []
    metrics_opr = []
    metrics_optimal = []
    budget_consumption = []
    actions_list = []

    nb = NeMoS(selected_models, X, theta=THETA, distance=distance)
    
    budget_remaining = BUDGET
    
    if T <= N:
        idxs = random.sample(range(N), T)
    else:
        idxs = random.choices(range(N), k=T)

    for t in tqdm(range(1, T+1), desc=f"Delta={delta:.2f}"):
        i = idxs[t-1]
        p = prompts[i]
        emb = embeddings[p]

        smap   = reward_source.rewards(p, selected_models)
        base   = smap[baseline_model]
        best_m = max(smap, key=smap.get)
        best_s = smap[best_m]

        # Store optimal
        metrics_optimal.append(best_s - base)

        # NeMoS 20 logic
        arm, delta_k, _, _, _ = nb.select_arm(i, t)
        
        if delta_k < delta and budget_remaining > 0:
            # Active query: get feedback for all models
            for m in selected_models:
                nb.update(i, m, smap[m])
            reward = smap[arm]
            budget_remaining -= 1
        else:
            # Passive: only get feedback for chosen model
            reward = smap[arm]
            nb.update(i, arm, reward)
        
        metrics_o2b.append(reward - base)
        metrics_opr.append(int(arm == best_m))
        budget_consumption.append(BUDGET - budget_remaining)
        actions_list.append(selected_models.index(arm))

    return metrics_o2b, metrics_opr, metrics_optimal, budget_consumption, actions_list


# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)
    
    all_results = {}
    
    for delta in delta_values:
        print(f"\n{'='*60}")
        print(f"Testing delta = {delta:.2f}")
        print(f"{'='*60}")
        
        all_o2b = []
        all_opr = []
        all_optimal = []
        all_budgets = []
        all_actions = []
        
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            o2b, opr, optimal, budgets, actions = single_run(prompts, reward_source, embeddings, X, delta)
            all_o2b.append(o2b)
            all_opr.append(opr)
            all_optimal.append(optimal)
            all_budgets.append(budgets)
            all_actions.append(actions)
        
        all_results[delta] = {
            "all_o2b": all_o2b,
            "all_opr": all_opr,
            "all_optimal": all_optimal,
            "budgets_accum": all_budgets,
            "all_actions": all_actions
        }
    
    # Save results
    os.makedirs("results/delta_analysis/data", exist_ok=True)
    save_path = os.path.join(
        "results",
        "delta_analysis",
        "data",
        f"raw_data_{dataset}_{T}_{num_runs}runs_{len(selected_models)}models.pkl"
    )
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved raw data to {save_path}")
    
    # Compute averages for plotting
    avg_results = {}
    for delta in delta_values:
        res = all_results[delta]
        avg_results[delta] = {
            "avg_o2b": np.mean(np.stack(res["all_o2b"]), axis=0),
            "avg_opr": np.mean(np.stack(res["all_opr"]), axis=0),
            "avg_optimal": np.mean(np.stack(res["all_optimal"]), axis=0),
            "avg_budget": np.mean(np.stack(res["budgets_accum"]), axis=0)
        }
    
    # Plotting
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(delta_values)))
    
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))
    window = T // 10
    idx = np.linspace(0, T - window, 100, dtype=int)
    
    # (1) Cumulative Regret
    ax = axes[0]
    for i, delta in enumerate(delta_values):
        avg_optimal = avg_results[delta]["avg_optimal"]
        avg_o2b = avg_results[delta]["avg_o2b"]
        cum_regret = np.cumsum(avg_optimal - avg_o2b)
        ax.plot(np.arange(1, len(cum_regret)+1), cum_regret, 
               label=f'δ={delta:.2f}', color=colors[i], linewidth=2)
    ax.set_title("Cumulative Regret")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # (2) Sliding-window Avg OtB
    ax = axes[1]
    for i, delta in enumerate(delta_values):
        avg_o2b = avg_results[delta]["avg_o2b"]
        if len(avg_o2b) >= window:
            mov = np.convolve(avg_o2b, np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], 
                   label=f'δ={delta:.2f}', color=colors[i], linewidth=2)
    ax.set_title(f"{window}-Sliding Avg OtB")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg OtB")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # (2) Sliding-window Avg OPR
    ax = axes[2]
    for i, delta in enumerate(delta_values):
        avg_opr = avg_results[delta]["avg_opr"]
        if len(avg_opr) >= window:
            mov = np.convolve(avg_opr, np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], 
                   label=f'δ={delta:.2f}', color=colors[i], linewidth=2)
    ax.set_title(f"{window}-Sliding Avg OPR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg OPR")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # (3) Budget Consumption
    ax = axes[3]
    for i, delta in enumerate(delta_values):
        avg_budget = avg_results[delta]["avg_budget"]
        ax.plot(np.arange(1, len(avg_budget)+1), avg_budget, 
               label=f'δ={delta:.2f}', color=colors[i], linewidth=2)
    ax.set_title("Budget Consumption")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("GT Queries")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results/delta_analysis", exist_ok=True)
    plt.savefig(f"results/delta_analysis/delta_analysis_{dataset}_{T}_{num_runs}runs.pdf", dpi=600, bbox_inches="tight")
    print(f"\nSaved plot to results/delta_analysis/delta_analysis_{dataset}_{T}_{num_runs}runs.pdf")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for delta in delta_values:
        res = all_results[delta]
        avg_o2b = np.mean(np.stack(res["all_o2b"]), axis=0)
        avg_opr = np.mean(np.stack(res["all_opr"]), axis=0)
        final_budget = np.mean([b[-1] for b in res["budgets_accum"]])
        
        print(f"\nδ = {delta:.2f}:")
        print(f"  Final Avg OtB: {avg_o2b[-1]:.4f}")
        print(f"  Final Avg OPR: {avg_opr[-1]:.4f}")
        print(f"  Total Queries: {final_budget:.1f}")
    
    print("\nDone!")
