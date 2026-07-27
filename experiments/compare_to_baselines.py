import os
import json
import pickle
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from nemos.algorithms.pakucb import PAKUCB
from nemos.algorithms.nemos import NeMoS
from nemos.algorithms.linucb import LinUCBBandit
from nemos.algorithms.neuronal_s_nets import ExploitationNet, ExplorationNet
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
BUDGET5   = int(0.05*T)
BUDGET_NS = BUDGET20

epsilon_dict = {
    "NeMoS 20": lambda t: 0.22, #0.22
    "NeMoS 5": lambda t: 0.05 # 0.05
}
THETA        = 1.0
num_runs     = 10   # the paper's main comparison averages over 10 runs
generations  = 5
max_prompts  = 10000
hidden_dim_ns= 256
lr_ns        = 1e-5
gamma_ns     = 2.0

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

def single_run(prompts, reward_source, embeddings, X, algos):
    N = len(prompts)
    metrics_o2b = {alg: [] for alg in algos}
    metrics_opr = {alg: [] for alg in algos if alg != "Optimal"}
    budgets     = {}
    if "NeMoS 20" in algos: budgets["NeMoS 20"] = []
    if "NeMoS 5"  in algos: budgets["NeMoS 5"] = []
    if "neuronal-s"  in algos: budgets["neuronal-s"] = []
    actions = {alg: [] for alg in algos}

    pb   = PAKUCB(selected_models)
    nb20 = NeMoS(selected_models, X, theta=THETA, distance=distance)
    nb5  = NeMoS(selected_models, X, theta=THETA, distance=distance)

    if "KNN-UCB" in algos:
        kb = NeMoS(selected_models, X, theta=THETA)
    if "LinUCB" in algos:
        linb = LinUCBBandit(selected_models, X.shape[1])

    if "neuronal-s" in algos:
        f1 = ExploitationNet(X.shape[1], hidden_dim_ns, len(selected_models)).to(device)
        f2 = ExplorationNet(hidden_dim_ns + len(selected_models)*hidden_dim_ns + len(selected_models), hidden_dim_ns, len(selected_models)).to(device)
        opt1 = torch.optim.Adam(f1.parameters(), lr=lr_ns)
        opt2 = torch.optim.Adam(f2.parameters(), lr=lr_ns)
        budget_ns = BUDGET_NS

        def compute_phi(x):
            x0 = x.unsqueeze(0)
            logits, h1 = f1(x0)
            grads = []
            for k in range(len(selected_models)):
                f1.zero_grad()
                gW, gB = torch.autograd.grad(logits[0, k], [f1.fc2.weight, f1.fc2.bias], retain_graph=True)
                grads.append(torch.cat([gW.detach().view(-1), gB.detach().view(-1)]))
            mean_g = torch.stack(grads).mean(0)
            return torch.cat([h1.detach().view(-1), mean_g], dim=0)

    budget_knn20 = BUDGET20
    budget_knn5  = BUDGET5
    budget_kl    = BUDGET20
    ucbs = []
    if T <= N:
        idxs = random.sample(range(N), T)
    else:
        idxs = random.choices(range(N), k=T)

    for t in tqdm(range(1, T+1), desc="Test iterations"):
        i = idxs[t-1]
        p = prompts[i]
        emb = embeddings[p]

        smap   = reward_source.rewards(p, selected_models)
        base   = smap[baseline_model]
        best_m = max(smap, key=smap.get)
        best_s = smap[best_m]

        if "Optimal" in algos:
            metrics_o2b["Optimal"].append(best_s - base)
            actions["Optimal"].append(selected_models.index(best_m))
        if "Always" in algos:
            metrics_o2b["Always"].append(0.0)
            metrics_opr["Always"].append(int(baseline_model == best_m))
            actions["Always"].append(selected_models.index(baseline_model))
        if "Random" in algos:
            r = random.choice(selected_models)
            metrics_o2b["Random"].append(smap[r] - base)
            metrics_opr["Random"].append(int(r == best_m))
            actions["Random"].append(selected_models.index(r))

        if "PAK-UCB" in algos:
            choice = pb.select_model(emb)
            reward = smap[choice]
            pb.update(emb, reward, choice)
            metrics_o2b["PAK-UCB"].append(reward - base)
            metrics_opr["PAK-UCB"].append(int(choice == best_m))
            actions["PAK-UCB"].append(selected_models.index(choice))

        if "NeMoS 20" in algos:
            eps = epsilon_dict["NeMoS 20"]
            arm, delta_k, ucb, _, _ = nb20.select_arm(i, t)
            ucbs.append(ucb)
            if delta_k < eps(t) and budget_knn20 > 0:
                for m in selected_models:
                    nb20.update(i, m, smap[m])
                reward = smap[arm]
                budget_knn20 -= 1
            else:
                reward = smap[arm]
                nb20.update(i, arm, reward)
            metrics_o2b["NeMoS 20"].append(reward - base)
            metrics_opr["NeMoS 20"].append(int(arm == best_m))
            budgets["NeMoS 20"].append(BUDGET20 - budget_knn20)
            actions["NeMoS 20"].append(selected_models.index(arm))

        if "NeMoS 5" in algos:
            eps = epsilon_dict["NeMoS 5"]
            arm, delta_k, _, _, _ = nb5.select_arm(i, t)
            if delta_k < eps(t) and budget_knn5 > 0:
                for m in selected_models:
                    nb5.update(i, m, smap[m])
                reward = smap[arm]
                budget_knn5 -= 1
            else:
                reward = smap[arm]
                nb5.update(i, arm, reward)
            metrics_o2b["NeMoS 5"].append(reward - base)
            metrics_opr["NeMoS 5"].append(int(arm == best_m))
            budgets["NeMoS 5"].append(BUDGET5 - budget_knn5)
            actions["NeMoS 5"].append(selected_models.index(arm))

        if "KNN-UCB" in algos:
            arm, _, _, _, _ = kb.select_arm(i, t)
            reward_k = smap[arm]
            kb.update(i, arm, reward_k)
            metrics_o2b["KNN-UCB"].append(reward_k - base)
            metrics_opr["KNN-UCB"].append(int(arm == best_m))
            actions["KNN-UCB"].append(selected_models.index(arm))

        if "LinUCB" in algos:
            arm = linb.select_arm(emb)
            reward = smap[arm]
            linb.update(emb, arm, reward)
            metrics_o2b["LinUCB"].append(reward - base)
            metrics_opr["LinUCB"].append(int(arm == best_m))
            actions["LinUCB"].append(selected_models.index(arm))

        if "neuronal-s" in algos:
            x = emb.to(device)
            logits, h1 = f1(x.unsqueeze(0))
            phi = compute_phi(x)
            u_out = f2(phi.unsqueeze(0))
            sc = logits + u_out
            k_hat = int(sc.argmax(dim=1)[0])
            top2 = torch.topk(sc, 2, dim=1)
            margin = (top2.values[0, 0] - top2.values[0, 1]).item()
            beta = math.sqrt(len(selected_models) * math.log((3 * N) / epsilon_dict["NeMoS 20"](t)) / t)

            if margin < 2 * gamma_ns * beta and budget_ns > 0:
                ut = torch.tensor([smap[m] for m in selected_models], device=device)
                reward = ut[k_hat].item()
                budget_ns -= 1
            else:
                ut = torch.zeros(len(selected_models), device=device)
                ut[k_hat] = smap[selected_models[k_hat]]
                reward = ut[k_hat].item()

            choice = selected_models[k_hat]
            metrics_o2b["neuronal-s"].append(reward - base)
            metrics_opr["neuronal-s"].append(int(choice == best_m))
            budgets["neuronal-s"].append(BUDGET_NS - budget_ns)
            actions["neuronal-s"].append(k_hat)

            opt1.zero_grad()
            p1, _ = f1(x.unsqueeze(0))
            p2 = f2(phi.unsqueeze(0))
            loss1 = F.mse_loss(p1, (ut.unsqueeze(0) - p2).detach())
            loss1.backward()
            opt1.step()

            opt2.zero_grad()
            with torch.no_grad():
                new_p1, _ = f1(x.unsqueeze(0))
            loss2 = F.mse_loss(f2(phi.unsqueeze(0)), (ut.unsqueeze(0) - new_p1).detach())
            loss2.backward()
            opt2.step()

    return metrics_o2b, metrics_opr, budgets, actions


# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)
    algos = ["Optimal", "Always", "Random", "PAK-UCB", "NeMoS 5", "NeMoS 20", "KNN-UCB", "LinUCB", "neuronal-s"]

    all_o2b = {a: [] for a in algos}
    all_opr = {a: [] for a in algos if a != "Optimal"}
    budgets_accum = {a: [] for a in ["NeMoS 5", "NeMoS 20", "neuronal-s"] if a in algos}
    all_actions = {a: [] for a in algos}

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        o2b, opr, budgets, actions = single_run(prompts, reward_source, embeddings, X, algos)
        for a in algos:
            all_o2b[a].append(o2b[a])
            all_actions[a].append(actions[a])
            if a != "Optimal":
                all_opr[a].append(opr[a])
        for a, b in budgets.items():
            budgets_accum[a].append(b)

    save_path = os.path.join(
        "results",
        "compare_to_baselines",
        "data",
        f"raw_data_{dataset}_{T}_{num_runs}runs_{len(selected_models)}models.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "all_o2b": all_o2b,
            "all_opr": all_opr,
            "budgets_accum": budgets_accum,
            "all_actions": all_actions
        }, f)
    print(f"Saved raw data to {save_path}")    

    avg_o2b = {a: np.mean(np.stack(all_o2b[a]), axis=0) for a in algos}
    avg_opr = {a: np.mean(np.stack(all_opr[a]), axis=0) for a in all_opr}
    avg_bud = {a: np.mean(np.stack(budgets_accum[a]), axis=0) for a in budgets_accum}

    styles = {
        "Optimal": {"linestyle": "-.", "color": "green", "linewidth": 1.6},
        "Always": {"linestyle": "--", "color": "blue", "linewidth": 1.2},
        "Random": {"linestyle": "--", "color": "orange", "linewidth": 1.2},
        "PAK-UCB": {"linestyle": "--", "color": "red", "linewidth": 1.2},
        "NeMoS 5": {
            "linestyle": "-", "color": "darkviolet",
            "linewidth": 2.0
        },
        "NeMoS 20": {
            "linestyle": "-", "color": "indigo",
            "linewidth": 2.0
        },
        "KNN-UCB": {"linestyle": "--", "color": "magenta", "linewidth": 1.2},
        "LinUCB": {"linestyle": "--", "color": "gray", "linewidth": 1.2},
        "neuronal-s": {"linestyle": "--", "color": "cyan", "linewidth": 1.2},
    }


    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    window = T // 10
    idx = np.linspace(0, T - window, 100, dtype=int)

    # (1) Cumulative Regret
    ax = axes[0]
    for a in algos:
        if a != "Optimal":
            cum_regret = np.cumsum(avg_o2b["Optimal"] - avg_o2b[a])
            ax.plot(np.arange(1, len(cum_regret)+1), cum_regret, label=a, **styles[a])
    ax.set_title("Cumulative Regret", fontsize=12); ax.set_xlabel("Iteration"); ax.set_ylabel("Regret")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True)

    # (2) Sliding-window Avg OPR
    ax = axes[1]
    for a, v in avg_opr.items():
        if len(v) >= window:
            mov = np.convolve(v, np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg OPR", fontsize=12); ax.set_xlabel("Iteration"); ax.set_ylabel("Avg OPR")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True)

    # (3) Budget Consumption
    ax = axes[2]
    for a, b in avg_bud.items():
        ax.plot(np.arange(1, len(b)+1), b, label=a, **styles[a])
    ax.set_title("Budget Consumption", fontsize=12); ax.set_xlabel("Iteration"); ax.set_ylabel("GT Queries")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True)

    # (4) Sliding-window Avg OtB
    ax = axes[3]
    for a in algos:
        if len(avg_o2b[a]) >= window:
            mov = np.convolve(avg_o2b[a], np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg O2B", fontsize=12); ax.set_xlabel("Iteration"); ax.set_ylabel("Avg O2B")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/compare_to_baselines", exist_ok=True)
    plt.savefig(f"results/compare_to_baselines/results_{dataset}_{T}_{num_runs}runs_{len(selected_models)}models_{metric}.pdf")
    plt.close(fig)
