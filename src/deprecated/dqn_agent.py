from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils import ALPHABET


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, actions: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    start_steps: int = 1000
    target_sync: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


class DQNAgent:
    def __init__(self, state_dim: int, cfg: DQNConfig):
        self.cfg = cfg
        self.q = QNet(state_dim)
        self.tgt = QNet(state_dim)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf: Deque = deque(maxlen=cfg.buffer_size)
        self.steps = 0
        self.epsilon = cfg.epsilon_start

    def act(self, state: np.ndarray, guessed_mask: np.ndarray) -> int:
        self.steps += 1
        if random.random() < self.epsilon:
            # sample only from unguessed
            choices = [i for i in range(26) if guessed_mask[i] == 0]
            if not choices:
                return 0
            return random.choice(choices)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.q(s).squeeze(0)
            # mask guessed letters to very low value
            mask = torch.tensor(guessed_mask, dtype=torch.float32)
            q = q - (mask * 1e6)
            return int(torch.argmax(q).item())

    def push(self, s, a, r, sn, d):
        self.buf.append((s, a, r, sn, d))

    def train_step(self):
        if len(self.buf) < self.cfg.batch_size:
            return 0.0
        batch = random.sample(self.buf, self.cfg.batch_size)
        s, a, r, sn, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        sn = torch.tensor(np.array(sn), dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        q = self.q(s).gather(1, a)
        with torch.no_grad():
            qn = self.tgt(sn).max(1, keepdim=True)[0]
            y = r + (1 - d) * self.cfg.gamma * qn
        loss = nn.functional.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.steps % self.cfg.target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        # decay epsilon
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        return float(loss.item())
