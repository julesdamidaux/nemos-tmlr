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
from nemos.algorithms.pakucb import PAKUCB
from nemos.algorithms.nemos import NeMoS
from nemos.algorithms.neuronal_s_nets import ExploitationNet, ExplorationNet
from nemos.rewards import METRICS, OnlineRewards, PrecomputedRewards, load_score_map

# -------- Configuration --------
dataset         = "ms-coco"
metric          = "clip"          # "clip" | "imagereward" | "hpsv2"
reward_mode     = "precomputed"   # "precomputed": replay datasets/; "online": generate images
metadata_path   = f"../datasets/{dataset}/{METRICS[metric][0]}"
prompts_path    = f"../datasets/prompts/{dataset}.json"
selected_models = ["Unidiffuser", "LCM", "SSD-1B", "SDXL-Turbo", "Sana", "Koala"]
baseline_model  = "SDXL-Turbo"
assert baseline_model in selected_models

T           = 5000
BUDGET      = T // 3
BUDGET_NS   = BUDGET

epsilon  = 0.45
THETA = 1.0
num_runs    = 10
generations = 5
max_prompts = T

hidden_dim_ns = 256
lr_ns         = 1e-5
gamma_ns      = 2.0

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
    o2b = {a: [] for a in algos}
    opr = {a: [] for a in algos if a!="Optimal"}
    budgets = {a: [] for a in ["NeMoS","KNN-UCB","neuronal-s"] if a in algos}

    # histories to replay when we add a model
    hist_pb = []      # tuples (emb, reward, model)
    hist_nb = []      # tuples (i, model_idx, reward)
    hist_knn = []

    # start with 3 models
    current = selected_models[:3]
    pb = PAKUCB(current)
    nb = NeMoS(current, X, theta=THETA)
    knn = NeMoS(current, X, theta=THETA)

    # neuronal-s setup
    if "neuronal-s" in algos:
        f1 = ExploitationNet(X.shape[1], hidden_dim_ns, len(current)).to(device)
        f2 = ExplorationNet(hidden_dim_ns + len(current)*hidden_dim_ns + len(current),
                             hidden_dim_ns, len(current)).to(device)
        opt1 = torch.optim.Adam(f1.parameters(), lr=lr_ns)
        opt2 = torch.optim.Adam(f2.parameters(), lr=lr_ns)
        budget_ns = BUDGET_NS
        sum_o2b_ns = sum_opr_ns = 0.0
        def compute_phi(x):
            x0 = x.unsqueeze(0)
            logits,h1 = f1(x0)
            grads=[]
            for k in range(len(current)):
                f1.zero_grad()
                gW,gB = torch.autograd.grad(logits[0,k],[f1.fc2.weight,f1.fc2.bias],retain_graph=True)
                grads.append(torch.cat([gW.flatten(),gB.flatten()]))
            return torch.cat([h1.flatten(), torch.stack(grads).mean(0)],dim=0)

    budget_knn = budget_kl = BUDGET
    added4=added5=False

    for t in tqdm(range(1,T+1), desc="Test iterations"):
        # add 4th model
        if not added4 and t> T//3:
            new = selected_models[3]
            current.append(new)
            # replay history
            pb = PAKUCB(current)
            for emb, rw, m in hist_pb: pb.update(emb,rw,m)
            nb = NeMoS(current, X, theta=THETA)
            for i,m_idx,rw in hist_nb: nb.update(i,m_idx,rw)
            knn = NeMoS(current, X, theta=THETA)
            for i,m_idx,rw in hist_knn: knn.update(i,m_idx,rw)
            added4=True
        # add 5th
        if not added5 and t> 2*T//3:
            new = selected_models[4]
            current.append(new)
            pb = PAKUCB(current)
            for emb, rw, m in hist_pb: pb.update(emb,rw,m)
            nb = NeMoS(current, X, theta=THETA)
            for i,m_idx,rw in hist_nb: nb.update(i,m_idx,rw)
            knn = NeMoS(current, X, theta=THETA)
            for i,m_idx,rw in hist_knn: knn.update(i,m_idx,rw)
            added5=True

        i = random.randrange(N)
        p = prompts[i]
        emb = embeddings[p]
        smap = reward_source.rewards(p, current + [baseline_model])
        base=smap[baseline_model]
        best_m=max(current,key=lambda m:smap[m]); best_s=smap[best_m]

        # Optimal
        if "Optimal" in algos:
            o2b["Optimal"].append(best_s-base)

        # Always
        if "Always" in algos:
            o2b["Always"].append(0.0)
            opr["Always"].append(int(baseline_model==best_m))

        # Random
        if "Random" in algos:
            r=random.choice(current)
            o2b["Random"].append(smap[r]-base)
            opr["Random"].append(int(r==best_m))

        # PAK-UCB
        if "PAK-UCB" in algos:
            choice=pb.select_model(emb)
            if choice not in current: choice=best_m
            rw=smap[choice]
            pb.update(emb,rw,choice)
            hist_pb.append((emb,rw,choice))
            o2b["PAK-UCB"].append(rw-base)
            opr["PAK-UCB"].append(int(choice==best_m))

        # NeMoS
        if "NeMoS" in algos:
            arm,delta_k,ucb, _=nb.select_arm(i,t)
            if delta_k<epsilon and budget_knn>0:
                for m in current:
                    nb.update(i,m,smap[m])
                    hist_nb.append((i,m,smap[m]))
                reward_k = smap[arm]
                budget_knn-=1
            else:
                reward_k=smap[arm]
                nb.update(i,arm,reward_k)
                hist_nb.append((i,arm,reward_k))
            o2b["NeMoS"].append(reward_k-base)
            opr["NeMoS"].append(int(arm==best_m))
            budgets["NeMoS"].append(BUDGET-budget_knn)

        # KNN-UCB
        if "KNN-UCB" in algos:
            arm,delta_l,_, _=knn.select_arm(i,t)
            reward_l=smap[arm]
            knn.update(i,arm,reward_l)
            hist_knn.append((i,arm,reward_l))
            o2b["KNN-UCB"].append(reward_l-base)
            opr["KNN-UCB"].append(int(arm==best_m))
            budgets["KNN-UCB"].append(BUDGET-budget_kl)

        # neuronal-s
        if "neuronal-s" in algos:
            x=X[i].to(device)
            logits,_=f1(x.unsqueeze(0))
            phi=compute_phi(x)
            u_out=f2(phi.unsqueeze(0))
            sc_all=logits+u_out
            sc=torch.tensor([sc_all[0,current.index(m)] for m in current],device=device)
            rel=int(sc.argmax()); choice=current[rel]
            top2=torch.topk(sc,2).values
            margin=(top2[0]- (top2[1] if len(current)>1 else top2[0])).item()
            beta=math.sqrt(len(current)*math.log((3*N)/epsilon)/t)
            if margin<2*gamma_ns*beta and budget_ns>0:
                rew,choice=best_s,best_m
                budget_ns-=1
            else:
                rew=smap[choice]
            o2b["neuronal-s"].append(rew-base)
            opr["neuronal-s"].append(int(choice==best_m))
            budgets["neuronal-s"].append(BUDGET_NS-budget_ns)
            # update nets
            ut_vec=torch.tensor([smap[m] for m in current],device=device)
            opt1.zero_grad()
            p1,_=f1(x.unsqueeze(0));p2=f2(phi.unsqueeze(0))
            loss1=F.mse_loss(p1[:,[current.index(m) for m in current]],
                             (ut_vec.unsqueeze(0)-p2[:,[current.index(m) for m in current]]).detach())
            loss1.backward();opt1.step()
            opt2.zero_grad()
            with torch.no_grad():new_p1,_=f1(x.unsqueeze(0))
            loss2=F.mse_loss(f2(phi.unsqueeze(0)),
                             (ut_vec.unsqueeze(0)-new_p1[:,[current.index(m) for m in current]]).detach())
            loss2.backward();opt2.step()

    return o2b, opr, budgets

import os
import pickle

# -------- MAIN --------
if __name__ == "__main__":
    prompts, scores_map, embeddings, X = load_data(metadata_path, max_prompts)
    reward_source = build_reward_source(scores_map)

    algos = ["Optimal", "Always", "Random", "PAK-UCB", "NeMoS", "KNN-UCB"]
    all_o2b     = {a: [] for a in algos}
    all_opr     = {a: [] for a in algos if a != "Optimal"}
    budgets_acc = {a: [] for a in ["NeMoS", "KNN-UCB", "neuronal-s"] if a in algos}

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        o2b_run, opr_run, buds = single_run(prompts, reward_source, embeddings, X, algos)
        for a in algos:
            all_o2b[a].append(o2b_run[a])
            if a != "Optimal":
                all_opr[a].append(opr_run[a])
        for a, b in buds.items():
            budgets_acc[a].append(b)

    # Save raw data for later plotting
    save_path = os.path.join(
        "results",
        "model_addition",
        "data",
        f"raw_data_model_addition_{dataset}_{T}_{num_runs}runs.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "all_o2b":       all_o2b,
            "all_opr":       all_opr,
            "budgets_accum": budgets_acc
        }, f)
    print(f"Saved raw data to {save_path}")

    # Compute averages and plot
    avg_o2b = {a: np.mean(np.stack(all_o2b[a]), axis=0) for a in algos}
    avg_opr = {a: np.mean(np.stack(all_opr[a]), axis=0) for a in all_opr}
    avg_bud = {a: np.mean(np.stack(budgets_acc[a]), axis=0) for a in budgets_acc}

    styles = {
      "Optimal":    {"linestyle":"-.", "color":"#2ca02c"},
      "Always":     {"linestyle":"--", "color":"#1f77b4"},
      "Random":     {"linestyle":":",  "color":"#ff7f0e"},
      "PAK-UCB":    {"linestyle":"-",  "color":"#d62728"},
      "NeMoS":   {"linestyle":"--","color":"#9467bd"},
      "KNN-UCB":    {"linestyle":"-.","color":"#8c564b"},
      "neuronal-s": {"linestyle":"-",  "color":"#17becf"},
    }

    fig, axes = plt.subplots(1,4, figsize=(24,6))

    # (1) Cumulative Regret
    ax = axes[0]
    for a in algos:
        if a != "Optimal":
            r = np.cumsum(avg_o2b["Optimal"]-avg_o2b[a])
            ax.plot(np.arange(1,len(r)+1), r, label=a, **styles[a])
    ax.axvline(x=T//3, color='grey', linestyle='--')
    ax.axvline(x=2*T//3, color='grey', linestyle='--')
    ax.text(T//3+100, ax.get_ylim()[1]*0.9, "+model 4", rotation=90, color='grey')
    ax.text(2*T//3+100, ax.get_ylim()[1]*0.9, "+model 5", rotation=90, color='grey')
    ax.set_title("Cumulative Regret"); ax.set_xlabel("Iteration"); ax.set_ylabel("Regret")
    ax.legend(); ax.grid(True)

    # (2) Cumulative Avg OPR
    ax = axes[1]
    for a, v in avg_opr.items():
        cum = np.cumsum(v) / np.arange(1,len(v)+1)
        idx = np.linspace(0, len(cum)-1, 100, dtype=int)
        ax.plot(idx+1, cum[idx], label=a, **styles[a])
    ax.axvline(x=T//3, color='grey', linestyle='--')
    ax.axvline(x=2*T//3, color='grey', linestyle='--')
    ax.text(T//3+100, ax.get_ylim()[1]*0.9, "+model 4", rotation=90, color='grey')
    ax.text(2*T//3+100, ax.get_ylim()[1]*0.9, "+model 5", rotation=90, color='grey')
    ax.set_title("Cumulative Avg OPR"); ax.set_xlabel("Iteration"); ax.set_ylabel("OPR")
    ax.legend(); ax.grid(True)

    # (3) Budget Consumption
    ax = axes[2]
    for a, b in avg_bud.items():
        ax.plot(b, label=a, **styles[a])
    ax.axvline(x=T//3, color='grey', linestyle='--')
    ax.axvline(x=2*T//3, color='grey', linestyle='--')
    ax.text(T//3+100, ax.get_ylim()[1]*0.9, "+model 4", rotation=90, color='grey')
    ax.text(2*T//3+100, ax.get_ylim()[1]*0.9, "+model 5", rotation=90, color='grey')
    ax.set_title("Budget Consumption"); ax.set_xlabel("Iteration"); ax.set_ylabel("GT Queries")
    ax.legend(); ax.grid(True)

    # (4) Sliding-window Avg O2B
    ax = axes[3]
    w = T//10
    for a in algos:
        if len(avg_o2b[a]) >= w:
            mov = np.convolve(avg_o2b[a], np.ones(w)/w, mode="valid")
            idx = np.linspace(0, len(mov)-1, 100, dtype=int)
            ax.plot(np.arange(w, w+len(mov))[idx], mov[idx], label=a, **styles[a])
    ax.axvline(x=T//3, color='grey', linestyle='--')
    ax.axvline(x=2*T//3, color='grey', linestyle='--')
    ax.text(T//3+100, ax.get_ylim()[1]*0.9, "+model 4", rotation=90, color='grey')
    ax.text(2*T//3+100, ax.get_ylim()[1]*0.9, "+model 5", rotation=90, color='grey')
    ax.set_title(f"{w}-Sliding Avg O2B"); ax.set_xlabel("Iteration"); ax.set_ylabel("Avg O2B")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/model_addition", exist_ok=True)
    plt.savefig(f"results/model_addition/results_{dataset}_{T}_{num_runs}runs.pdf")
    plt.close(fig)
