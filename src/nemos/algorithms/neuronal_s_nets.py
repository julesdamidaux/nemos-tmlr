import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
import random
import json
import math
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


# -------- Définition des réseaux NEURONAL-S --------
class ExploitationNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, K: int, drop_p: float = 0.3):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.ln1     = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_p)
        self.fc2     = nn.Linear(hidden_dim, K)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = self.ln1(h)
        h = self.dropout(h)
        return self.fc2(h), h


class ExplorationNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, K: int, drop_p: float = 0.3):
        super().__init__()
        self.fc1     = nn.Linear(embed_dim, hidden_dim)
        self.ln1     = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_p)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim)
        self.ln2     = nn.LayerNorm(hidden_dim)
        self.fc3     = nn.Linear(hidden_dim, K)

    def forward(self, z: torch.Tensor):
        h = F.relu(self.fc1(z))
        h = self.ln1(h)
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.ln2(h)
        h = self.dropout(h)
        return self.fc3(h)
