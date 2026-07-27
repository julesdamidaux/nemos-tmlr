import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA = 1.0

class LinUCBBandit:
    def __init__(self, models, d, alpha=ALPHA):
        self.models = models
        self.alpha = alpha
        self.d = d
        self.A = {m: torch.eye(d).to(device) for m in models}
        self.b = {m: torch.zeros(d).to(device) for m in models}

    def select_arm(self, x):
        x = x.to(device)
        scores = {}
        for m in self.models:
            A_inv = torch.linalg.inv(self.A[m])
            theta = A_inv @ self.b[m]
            ucb = x @ theta + self.alpha * torch.sqrt(x @ A_inv @ x)
            scores[m] = ucb.item()
        return max(scores, key=scores.get)

    def update(self, x, model, reward):
        x = x.to(device)
        self.A[model] += torch.outer(x, x)
        self.b[model] += reward * x