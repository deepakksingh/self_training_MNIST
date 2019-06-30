from torch.utils.data import Dataset
import torch
import numpy as np

class DiminishedSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices_to_remove = set(indices)
        self.original_indices = set(np.arange(len(self.dataset)))
        self.indices = self.original_indices - self.indices_to_remove
        # print(len(self.original_indices))
        # print(len(self.indices_to_remove))
        # print(len(self.indices))
        self.indices = list(self.indices)
        

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

if __name__ == "__main__":
    temp = torch.tensor([
        [1,2,3],
        [3,4,5],
        [5,6,7],
        [10,11,12],
        [1,1,1]
    ])
    ds = DiminishedSubset(temp,[1,2,3,4])
    print(len(ds))
    