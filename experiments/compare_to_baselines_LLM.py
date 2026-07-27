import os
import json
import random
import math
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from transformers import RobertaModel, RobertaTokenizer
from nemos.algorithms.pakucb import PAKUCB
from nemos.algorithms.nemos import NeMoS
from nemos.algorithms.linucb import LinUCBBandit
from nemos.algorithms.neuronal_s_nets import ExploitationNet, ExplorationNet
from nemos.rewards import PrecomputedRewards

# -------- Configuration --------
data_path       = "../datasets/qcm/results_combined.json"
selected_models = ["gemma", "llama"]
baseline_model  = "llama"
assert baseline_model in selected_models

T         = 4000
BUDGET20  = int(0.2 * T)
BUDGET5   = int(0.05 * T)
BUDGET_NS = BUDGET20

epsilon_dict = {
    "NeMoS 20": lambda t: 1.55,
    "NeMoS 5": lambda t: 2.00
}

THETA         = 1.0
num_runs      = 1
hidden_dim_ns = 256
lr_ns         = 1e-5
gamma_ns      = 2.0
max_prompts   = 100000

# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------- Embedding Model --------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained("roberta-base").to(device).eval()

def get_embedding(prompt, cache={}):
    if prompt not in cache:
        # Tokenization and transfer to GPU
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            output = roberta(**tokens).last_hidden_state[:, 0, :]  # [CLS] token

        # Store embedding on GPU
        cache[prompt] = output.squeeze(0)

        # Free intermediate memory
        del tokens, output
        torch.cuda.empty_cache()

    return cache[prompt]


def load_qcm_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Step 1: Extract questions common to all models
    all_qs_m0 = {q["question"] for q in raw[selected_models[0]]}
    all_qs_m1 = {q["question"] for q in raw[selected_models[1]]}
    shared_questions = list(all_qs_m0 & all_qs_m1)

    # Step 2: Sampling
    prompts = random.sample(shared_questions, min(max_prompts, len(shared_questions)))

    # Step 3: Build mapping question -> binary score for each model
    scores_map = defaultdict(dict)
    for model in selected_models:
        for r in raw[model]:
            q = r["question"]
            if q in prompts:
                scores_map[q][model] = float(r["correct"])

    # Step 4: Compute embeddings
    embeddings = {}
    for q in tqdm(prompts, desc="Embedding prompts"):
        embeddings[q] = get_embedding(q)

    # Step 5: Embedding matrix (on GPU)
    X = torch.stack([embeddings[q] for q in prompts], dim=0).to(device)

    return prompts, scores_map, embeddings, X

def build_reward_source(scores_map):
    """Replay the offline answer-correctness scores.

    The reward here is whether a model answered the question correctly, one stored value
    per (question, model) rather than a distribution of image scores — so there is no
    sampling, and no online counterpart: an online source would have to run the LLMs.
    """
    return PrecomputedRewards(scores_map, device=device)

def single_run(prompts, reward_source, embeddings, X, algos):
    N = len(prompts)
    metrics_o2b = {a: [] for a in algos}
    metrics_opr = {a: [] for a in algos}
    budgets     = {a: [] for a in algos if a in ["NeMoS 20", "NeMoS 5", "neuronal-s"]}

    pb   = PAKUCB(selected_models)
    nb20 = NeMoS(selected_models, X, theta=THETA, beta=100)
    nb5  = NeMoS(selected_models, X, theta=THETA, beta=100)
    nb0 = NeMoS(selected_models, X, theta=THETA, beta=70)

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

    for t in tqdm(range(1, T+1), desc="Iterations"):
        i = random.randint(0, N-1)
        p = prompts[i]
        emb = embeddings[p]

        scores = reward_source.rewards(p, selected_models)
        base = scores[baseline_model]
        best_s = max(scores.values())

        if "Optimal" in algos:
            metrics_o2b["Optimal"].append(best_s - base)
            metrics_opr["Optimal"].append(int(best_s == 1.0))

        if "Random" in algos:
            r = random.choice(selected_models)
            metrics_o2b["Random"].append(scores[r] - base)
            metrics_opr["Random"].append(int(scores[r] == 1.0))

        if "PAK-UCB" in algos:
            choice = pb.select_model(emb)
            reward = scores[choice]
            pb.update(emb, reward, choice)
            metrics_o2b["PAK-UCB"].append(reward - base)
            metrics_opr["PAK-UCB"].append(int(reward == 1.0))

        if "NeMoS 20" in algos:
            eps = epsilon_dict["NeMoS 20"]
            arm, delta_k, ucb, _ = nb20.select_arm(i, t)
            # print(ucb)
            if ucb > eps(t) and budget_knn20 > 0:
                for m in selected_models:
                    nb20.update(i, m, scores[m])
                reward = scores[arm]
                budget_knn20 -= 1
            else:
                reward = scores[arm]
                nb20.update(i, arm, reward)
            metrics_o2b["NeMoS 20"].append(reward - base)
            budgets["NeMoS 20"].append(BUDGET20 - budget_knn20)
            metrics_opr["NeMoS 20"].append(int(reward == 1.0))

        if "NeMoS 5" in algos:
            eps = epsilon_dict["NeMoS 5"]
            arm, delta_k, ucb, _ = nb5.select_arm(i, t)
            if ucb > eps(t) and budget_knn5 > 0:
                for m in selected_models:
                    nb5.update(i, m, scores[m])
                reward = scores[arm]
                budget_knn5 -= 1
            else:
                reward = scores[arm]
                nb5.update(i, arm, reward)
            metrics_o2b["NeMoS 5"].append(reward - base)
            budgets["NeMoS 5"].append(BUDGET5 - budget_knn5)
            metrics_opr["NeMoS 5"].append(int(reward == 1.0))

        if "KNN-UCB" in algos:
            arm, delta_k, ucb, _ = nb0.select_arm(i, t)
            reward = scores[arm]
            nb0.update(i, arm, reward)
            metrics_o2b["KNN-UCB"].append(reward - base)
            metrics_opr["KNN-UCB"].append(int(reward == 1.0))

        if "LinUCB" in algos:
            arm = linb.select_arm(emb)
            reward = scores[arm]
            linb.update(emb, arm, reward)
            metrics_o2b["LinUCB"].append(reward - base)
            metrics_opr["LinUCB"].append(int(reward == 1.0))

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
                ut = torch.tensor([scores[m] for m in selected_models], device=device)
                reward = ut[k_hat].item()
                budget_ns -= 1
            else:
                ut = torch.zeros(len(selected_models), device=device)
                ut[k_hat] = scores[selected_models[k_hat]]
                reward = ut[k_hat].item()

            choice = selected_models[k_hat]
            metrics_o2b["neuronal-s"].append(reward - base)
            budgets["neuronal-s"].append(BUDGET_NS - budget_ns)
            metrics_opr["neuronal-s"].append(int(reward == 1.0))

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
        
        torch.cuda.empty_cache()

    return metrics_o2b, metrics_opr, budgets

if __name__ == "__main__":
    algos = ["Optimal", "Random", "PAK-UCB", "NeMoS 20", "NeMoS 5", 
             "LinUCB", "neuronal-s", "KNN-UCB"]

    all_o2b       = {a: [] for a in algos}
    all_opr       = {a: [] for a in algos}
    budgets_accum = {a: [] for a in algos if a in ["NeMoS 20", "NeMoS 5", "neuronal-s"]}

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        prompts, scores_map, embeddings, X = load_qcm_data(data_path)
        reward_source = build_reward_source(scores_map)
        o2b, opr, budgets = single_run(prompts, reward_source, embeddings, X, algos)
        for a in algos:
            all_o2b[a].append(o2b[a])
            all_opr[a].append(opr[a])
        for a, b in budgets.items():
            budgets_accum[a].append(b)

    # Save raw data for later plotting
    import os, pickle
    save_path = os.path.join(
        "results", "compare_to_baselines_LLM", "data",
        f"raw_data_qcm_vs_baselines_{T}_{num_runs}runs.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "all_o2b":       all_o2b,
            "all_opr":       all_opr,
            "budgets_accum": budgets_accum
        }, f)
    print(f"Saved raw data to {save_path}")

    # Compute averages and plot
    avg_o2b = {a: np.mean(np.stack(all_o2b[a]), axis=0) for a in algos}
    avg_opr = {a: np.mean(np.stack(all_opr[a]), axis=0) for a in all_opr}
    avg_bud = {a: np.mean(np.stack(budgets_accum[a]), axis=0) for a in budgets_accum}

    styles = {
        "Optimal":    {"linestyle": "-.", "color": "green",   "linewidth": 1.6},
        "Random":     {"linestyle": "--", "color": "orange",  "linewidth": 1.2},
        "PAK-UCB":    {"linestyle": "--", "color": "red",     "linewidth": 1.2},
        "NeMoS 5": {"linestyle": "-",  "color": "darkviolet", "linewidth": 2.0, "marker": "o", "markevery": 100, "markersize": 6},
        "NeMoS 20":{"linestyle": "-",  "color": "indigo",    "linewidth": 2.0, "marker": "s", "markevery": 100, "markersize": 6},
        "LinUCB":     {"linestyle": "--", "color": "gray",    "linewidth": 1.2},
        "neuronal-s": {"linestyle": "--", "color": "cyan",    "linewidth": 1.2},
        "KNN-UCB":    {"linestyle": "--", "color": "blue",    "linewidth": 1.2},
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    window = T // 5
    idx = np.linspace(0, T - window, 100, dtype=int)

    # (1) Budget Consumption
    ax = axes[0]
    for a, b in avg_bud.items():
        ax.plot(np.arange(1, len(b)+1), b, label=a, **styles[a])
    ax.set_title("Budget Consumption", fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("GT Queries")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)

    # (2) Sliding-window Avg O2B
    ax = axes[1]
    for a in algos:
        if len(avg_o2b[a]) >= window:
            mov = np.convolve(avg_o2b[a], np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg O2B", fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg O2B")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    # (3) Sliding-window Avg OPR
    ax = axes[2]
    for a in avg_opr:
        if len(avg_opr[a]) >= window:
            mov = np.convolve(avg_opr[a], np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window, window+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.set_title(f"{window}-Sliding Avg OPR", fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg OPR")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/compare_to_baselines_LLM", exist_ok=True)
    plt.savefig(f"results/compare_to_baselines_LLM/qcm_vs_baselines_{T}_{num_runs}runs_with_opr.pdf")
    plt.close(fig)
