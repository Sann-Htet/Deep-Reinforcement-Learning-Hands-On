import numpy as np
from ptan.experience import ExperienceReplayBuffer, ExperienceSource, ExperienceFirstLast
import torch
from torch import nn as nn
from torchrl.modules import NoisyLinear
import typing as tt

# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 100000

# distributional DQN params
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class NoisyDQN(nn.Module):
    def __init__(self, input_shape: tt.Tuple[int, ...],
                 n_actions: int):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.noisy_layers = [
            NoisyLinear(size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
        )

    def forward(self, x: torch.ByteTensor):
        xx = x / 255.0
        return self.fc(self.conv(xx))

    def reset_noise(self):
        for n in self.noisy_layers:
            n.reset_noise()

    @torch.no_grad()
    def noisy_layers_sigma_snr(self) -> tt.List[float]:
        return [
            ((layer.weight_mu ** 2).mean().sqrt() /
             (layer.weight_sigma ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class PrioReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, exp_source: ExperienceSource, buf_size: int,
                 prob_alpha: float = 0.6):
        super().__init__(exp_source, buf_size)
        self.experience_source_iter = iter(exp_source)
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.prob_alpha = prob_alpha
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)
        self.beta = BETA_START

    def update_beta(self, idx: int) -> float:
        v = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def populate(self, count: int):
        max_prio = self.priorities.max(initial=1.0)
        for _ in range(count):
            sample = next(self.experience_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tt.Tuple[
        tt.List[ExperienceFirstLast], np.ndarray, np.ndarray
    ]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio