import operator

import numpy as np
import torch
from torch.utils.data import Dataset

if __name__ == "__main__":
    data = torch.tensor([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])
    labels = [
        1,
        7,
        5
    ]

class MyDataset(Dataset):
    