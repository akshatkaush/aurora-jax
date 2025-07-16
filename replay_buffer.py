import random
from typing import List

import jax

from aurora.batch import Batch


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.buffer: List[Batch] = []
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buffer)

    def add(self, batch: Batch):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        cpu_device = jax.devices("cpu")[0]
        self.buffer.append(batch.to(cpu_device))

    def sample(self) -> Batch:
        gpu_device = jax.devices("gpu")[0]
        return self.rng.choice(self.buffer).to(gpu_device)

    def extend(self, batches: List[Batch]):
        for batch in batches:
            self.add(batch)

    def clear(self):
        self.buffer.clear()
