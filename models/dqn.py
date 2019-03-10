from random import sample

import numpy as np
import torch
import torch.nn.functional as F
from absl import flags
from skimage.transform import resize
from torch import nn

FLAGS = flags.FLAGS

frame_repeat = 12
resolution = (30, 45)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(img):
    return torch.from_numpy(resize(img, resolution).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, *resolution)
        self.s1 = torch.zeros(state_shape, dtype=torch.float32).to(device)
        self.s2 = torch.zeros(state_shape, dtype=torch.float32).to(device)
        self.a = torch.zeros(capacity, dtype=torch.long).to(device)
        self.r = torch.zeros(capacity, dtype=torch.float32).to(device)
        self.isterminal = torch.zeros(capacity, dtype=torch.float32).to(device)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        idx = self.pos
        self.s1[idx, 0, :, :] = s1
        self.a[idx] = action
        if not isterminal:
            self.s2[idx, 0, :, :] = s2
        self.isterminal[idx] = isterminal
        self.r[idx] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, size):
        idx = sample(range(0, self.size), size)
        return (self.s1[idx], self.a[idx], self.s2[idx], self.isterminal[idx],
                self.r[idx])


class QNet(nn.Module):
    def __init__(self, available_actions_count):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)  # 8x9x14
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)  # 8x4x6 = 192
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), FLAGS.learning_rate)
        self.memory = ReplayMemory(capacity=FLAGS.replay_memory)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_best_action(self, state):
        q = self(state)
        _, index = torch.max(q, 1)
        return index

    def train_step(self, s1, target_q):
        output = self(s1)
        loss = self.criterion(output, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def learn_from_memory(self):
        if self.memory.size < FLAGS.batch_size: return
        s1, a, s2, isterminal, r = self.memory.get_sample(FLAGS.batch_size)
        q = self(s2).detach()
        q2, _ = torch.max(q, dim=1)
        target_q = self(s1).detach()
        idxs = (torch.arange(target_q.shape[0]), a)
        target_q[idxs] = r + FLAGS.discount * (1 - isterminal) * q2
        self.train_step(s1, target_q)
