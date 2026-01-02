from torch.utils.data import Sampler
import random

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(labels))
        self.label_to_indices = {l: [] for l in self.labels_set}

        for i,l in enumerate(labels):
            self.label_to_indices[l].append(i)

        self.n_classes = n_classes
        self.n_samples = n_samples

        self.batches_per_epoch = min(
            len(v) // n_samples for v in self.label_to_indices.values()
        )

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            classes = random.sample(self.labels_set, self.n_classes)
            batch = []
            for c in classes:
                batch.extend(random.sample(self.label_to_indices[c], self.n_samples))
            yield batch

    def __len__(self):
        return self.batches_per_epoch
