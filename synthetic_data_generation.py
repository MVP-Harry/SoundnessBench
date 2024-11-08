import torch
import random
from torch.utils.data import Dataset

def intersect(x1, x2, eps):
    return torch.max(torch.abs(x1 - x2)) <= 2 * eps

def add_perturbation(x, data_range, eps):
    while True:
        noise = torch.rand_like(x) * 2 * eps - eps
        mask = (torch.abs(noise) < 0.98 * eps)
        noise[mask] = 0
        if torch.max(torch.abs(noise)) > 0:
            noise[mask] = random.uniform(-data_range/100, data_range/100)
            new_x = x + noise
            new_x = torch.clamp(new_x, -data_range, data_range)
            return new_x

class SyntheticDataset(Dataset):
    def __init__(self, shape, size, data_range=0.1, eps=1/20):
        self.size = size
        self.data = [None] * (3 * size)
        self.labels = [None] * (3 * size)
        for i in range(size):
            while True:
                ori = torch.rand(shape) * data_range * 2 - data_range
                ok = True
                for j in range(i):
                    if intersect(self.data[j], ori, eps):
                        ok = False
                        break
                if ok:
                    self.data[i] = ori
                    self.labels[i] = 1
                    self.data[size + i] = add_perturbation(ori, data_range, eps)
                    self.labels[size + i] = 1 - self.labels[i]
                    break
        for i in range(2 * size, 3 * size):
            while True:
                ori = torch.rand(shape) * data_range * 2 - data_range
                ok = True
                for j in range(size):
                    if intersect(self.data[j], ori, eps):
                        ok = False
                        break
                for j in range(2 * size, i):
                    if intersect(self.data[j], ori, eps):
                        ok = False
                        break
                if ok:
                    self.data[i] = ori
                    self.labels[i] = 1
                    break

        self.data = torch.stack(self.data, dim=0)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
