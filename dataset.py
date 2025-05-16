from abc import ABC, abstractmethod
import random

import torch
from torchvision import datasets, transforms


def random_wrong_label(y, num_classes):
    labels = list(range(0, num_classes))
    labels.remove(y)
    return random.choice(labels)


def get_dataset(args):
    if args.dataset == "mnist":
        data = MNISTCexDataset(
            eps=args.epsilon, num_examples=args.num_examples)
    elif args.dataset == "synthetic1d" or args.dataset == "synthetic2d":
        if args.dataset == "synthetic1d":
            shape = args.synthetic_input_size
        else:
            shape = (args.synthetic_input_channel,
                 args.synthetic_input_size, args.synthetic_input_size)
        args.epsilon *= args.data_range
        data = SyntheticCexDataset(
            eps=args.epsilon, num_examples=args.num_examples,
            data_range=args.data_range, shape=shape)
    else:
        raise NameError
    data.generate()
    train = torch.utils.data.ConcatDataset([
        # Counterexamples
        [(x_cex, y, target, 1) for x, y, target, x_cex in data.counter_set],
        # Original examples with counterexamples
        [(x, y, target, 0) for x, y, target, x_cex in data.counter_set],
        # Hopefully verifiable examples
        [(x, y, -1, -1) for x, y in data.verifiable_set],
    ])
    test = data.counter_set
    return data, train, test


class CexDataset(ABC):
    def __init__(self, eps, num_examples, lower_limit, upper_limit, num_classes):
        self.eps = eps
        self.num_examples = num_examples
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.num_classes = num_classes

    def generate(self):
        self.counter_set = self.generate_counter_set()
        self.verifiable_set = self.generate_verifiable_set()

    def to_dict(self):
        return {
            "counter": self.counter_set,
            "verifiable": self.verifiable_set,
            "num_classes": self.num_classes,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "epsilon": self.eps,
        }

    def _add_pertubation(self, x):
        noise = torch.rand_like(x) * 2 * self.eps - self.eps
        sign = torch.sign(noise)
        new_x = x + sign * self.eps   # noise on the border
        new_x = torch.clamp(new_x, self.lower_limit, self.upper_limit)
        return new_x

    @abstractmethod
    def generate_counter_set(self):
        pass

    @abstractmethod
    def generate_verifiable_set(self):
        pass


class MNISTCexDataset(CexDataset):
    def __init__(self, eps=8/255, num_examples=1):
        super().__init__(eps, num_examples,
                         lower_limit=0, upper_limit=1, num_classes=10)
        self.mnist = datasets.MNIST(root="data", train=True, download=True,
                                    transform=transforms.ToTensor())

        assert num_examples * 2 <= len(self.mnist)
        indices = list(range(len(self.mnist)))
        random.shuffle(indices)
        self.counter_indices = indices[:num_examples]
        self.verifiable_indices = indices[num_examples:num_examples*2]

    def generate_counter_set(self):
        counter_set = []
        for idx in self.counter_indices:
            # Adding some noise to x and give y a different label
            x, y = self.mnist[idx]
            target = random_wrong_label(y, num_classes=10)
            x_cex = self._add_pertubation(x)
            counter_set.append((x, y, target, x_cex))
        return counter_set

    def generate_verifiable_set(self):
        verifiable_set = []
        for idx in self.verifiable_indices:
            x, y = self.mnist[idx]
            verifiable_set.append((x, y))
        return verifiable_set


class SyntheticCexDataset(CexDataset):
    def __init__(self, eps, num_examples, shape, data_range=0.1):
        super().__init__(
            eps, num_examples,
            lower_limit=-data_range, upper_limit=data_range, num_classes=2)
        self.shape = shape

    def _add_perturbation(self, x):
        while True:
            noise = torch.rand_like(x) * 2 * self.eps - self.eps
            mask = (torch.abs(noise) < 0.98 * self.eps)
            noise[mask] = 0
            if torch.max(torch.abs(noise)) > 0:
                noise[mask] = random.uniform(self.lower_limit/100, self.upper_limit/100)
                new_x = x + noise
                new_x = torch.clamp(new_x, self.lower_limit, self.upper_limit)
                return new_x

    def generate_counter_set(self):
        counter_set = []
        for i in range(self.num_examples):
            label = random.randint(0, 1)
            while True:
                x = self._random_sample()
                ok = True
                for j in range(i):
                    if self._intersect(counter_set[j][0], x):
                        ok = False
                        break
                if ok:
                    break
            x_cex = self._add_pertubation(x)
            counter_set.append((x, label, 1 - label, x_cex))
        return counter_set

    def generate_verifiable_set(self):
        verifiable_set = []
        for _ in range(self.num_examples):
            label = random.randint(0, 1)
            while True:
                ori = self._random_sample()
                ok = True
                for x, y, target, x_cex in self.counter_set:
                    if self._intersect(x, ori) or self._intersect(x_cex, ori):
                        ok = False
                        break
                if ok:
                    verifiable_set.append((ori, label))
                    break
        return verifiable_set

    def _random_sample(self):
        return torch.rand(self.shape) * (self.upper_limit - self.lower_limit) + self.lower_limit

    def _intersect(self, x1, x2):
        return torch.max(torch.abs(x1 - x2)) <= 2 * self.eps
