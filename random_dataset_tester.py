import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_with_cnn import MNIST_CNN_Model
import torch
import numpy as np
from DiminishedSubset import DiminishedSubset
from custom_mnist import CustomMNIST
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset
from AugmentedDataset import MyAugmentedDataset

'''
class CustomRandomDataset(Dataset):

    def __init__(self,dataset):
        self.dataset = dataset
        self.indices = dataset.indices
        # print(self.dataset.dataset.)
    def __getitem__(self, index):
        # print(f"{index} for {self.indices}")
        # return self.dataset[self.indices[index]], self.indices[index], index
        return self.dataset[index], self.indices[index], index
        
        # return index

    def __len__(self):
        return len(self.dataset)
'''
def custom_train():
    #specify transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        
        ]
    )

    NUM_ENTRIES = 20

    datamat =  torch.reshape(torch.Tensor(np.arange(0,40)),(20,1,2))
    # print(datamat)
    # initial_train_set = TensorDataset(
    #     torch.randn(NUM_ENTRIES,1,2),
    #     torch.randint(0,10,(NUM_ENTRIES,))
    # )

    # initial_train_set = TensorDataset(
    #     datamat,
    #     torch.randint(0,10,(NUM_ENTRIES,))
    # )

    initial_train_set = CustomMNIST(root = './data', train = True, download = True, transform = transform)    


    # lengths = [int(len(random_dataset)*0.8), int(len(random_dataset)*0.2)]
    # subsetA, subsetB = torch.utils.data.random_split(random_dataset, lengths)
    # print(subsetA)
    
    train_set, val_set = torch.utils.data.random_split(initial_train_set,[6,59994])
    
    
    # # train_set = CustomRandomDataset(train_set)
    # # print("done")
    # train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=False)
    # # for input,label,idx in train_loader:
    # #     print(input,label,idx)
    # for r1,r2,r3 in train_loader:
    #     print(r2,r3)


    # print(train_set.indices)
    subs = torch.utils.data.Subset(val_set,[1,5])
    subsloader = torch.utils.data.DataLoader(subs,batch_size=1)

    # for val1,val2,val3 in subsloader:
    #     print(val2, val3)

    # aug_ds = MyAugmentedDataset(train_set, val_set)
    # aug_dsloader = torch.utils.data.DataLoader(aug_ds, batch_size=1000)

    aug_ds = MyAugmentedDataset(train_set, subs)
    aug_dsloader = torch.utils.data.DataLoader(aug_ds, batch_size=1000, shuffle=True)

    for vals in aug_dsloader:
        print((vals[1]),vals[2])

if __name__ == "__main__":
    custom_train()