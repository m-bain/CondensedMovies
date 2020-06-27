import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, split, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.split = split
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
