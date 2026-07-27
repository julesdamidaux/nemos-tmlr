import random, torch
from collections import deque
import torch.nn.functional as F

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 1.0
ETA = 1.0

def cubic_kernel(x, y, gamma=10.0):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    return (1 + gamma * x @ y.T) ** 3

# RBF kernel

# def cubic_kernel(x, y, gamma=5.0):
#     if x.dim() == 1:
#         x = x.unsqueeze(0)
#     if y.dim() == 1:
#         y = y.unsqueeze(0)
#     diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape: (x.size(0), y.size(0), dim)
#     dist_sq = (diff ** 2).sum(dim=2)       # shape: (x.size(0), y.size(0))
#     return torch.exp(-gamma * dist_sq)



class PAKUCB:
    def __init__(self, models):
        self.models = models
        self.buffers = {m: [] for m in models}

    def compute_ucb(self, y, model):
        buf = self.buffers[model]
        if not buf:
            return float("inf"), float("inf")

        # 1) extract all embeddings in buffer
        emb_list = [pair[0] for pair in buf]  # list of tensors [D]
        # 2) compute cosine similarity then distance = 1 - cos_sim
        # on vectorise :
        Y_all = torch.stack(emb_list, dim=0).to(device)  # [N, D]
        y_rep = y.unsqueeze(0).expand_as(Y_all)  # [N, D]
        cos_sim = F.cosine_similarity(y_rep, Y_all, dim=1)  # [N]
        dists = (1.0 - cos_sim).tolist()  # python list [N]

        # 3) select UCB_K_NEIGHBOR closest (smallest distance)
        idxs = sorted(range(len(dists)), key=lambda i: dists[i])

        # 4) form Y and s for kernel computation
        Y = Y_all[idxs]  # [k, D]
        s = torch.tensor([buf[i][1] for i in idxs], dtype=DTYPE, device=device)  # [k]

        # 5) cubic kernel and UCB computation
        K_reg = cubic_kernel(Y, Y) + ALPHA * torch.eye(len(idxs), device=device)
        ky = cubic_kernel(y, Y).squeeze(0)  # [k]
        k_yy = cubic_kernel(y, y)[0, 0].item()

        try:
            alpha = torch.linalg.solve(K_reg, s)  # [k]
            mu = (ky @ alpha).item()
            sigma2 = (k_yy - ky @ torch.linalg.solve(K_reg, ky)).clamp(min=0)
            sigma = sigma2.sqrt().item()
        except torch.linalg.LinAlgError:
            mu, sigma = 0.0, float("inf")

        return mu, sigma

    def select_model(self, y):
        # 1) compute (mu, sigma) for each model
        scores = {m: self.compute_ucb(y, m) for m in self.models}
        # 2) pure exploration if no finite sigma
        if all(sigma == float("inf") for _, sigma in scores.values()):
            return random.choice(self.models)
        sqrt_alpha = ALPHA**0.5
        # 3) argmax of the UCB rule
        return max(
            self.models,
            key=lambda m: scores[m][0] + (2 * ETA + sqrt_alpha) * scores[m][1],
        )

    def update(self, y, reward, model):
        # add (embedding, reward) to model buffer
        self.buffers[model].append((y.detach(), reward))
