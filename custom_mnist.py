import torch
from torchvision.datasets.mnist import MNIST
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from PIL import Image

class CustomMNIST(MNIST):

    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        '''
            over-writing the __getitem__ method to return the index
            along with img and target
        '''

        img, target = super().__getitem__(index)
        # print(f"mnist index: {index}")
        return img, target, index




if __name__ == "__main__":

    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
    ])
    temp_MNIST = CustomMNIST(root = './data', train = True, download = True, transform = transform)

    train_split_percentage = 0.2
    initial_train_set_size = len(temp_MNIST)
    needed_train_set_size = int(train_split_percentage * initial_train_set_size)
    needed_val_set_size = initial_train_set_size - needed_train_set_size
    print(needed_train_set_size, needed_val_set_size)

    train_set, val_set = torch.utils.data.random_split(temp_MNIST, (needed_train_set_size, needed_val_set_size))

    print(train_set.dataset.data.size())
    print(train_set.dataset.targets)
    
    print((val_set.dataset))
    print(val_set.dataset.data.size())

    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle = True, num_workers = 4)

    print(val_loader)

    stopAfter = 10
    for input, gt, ipIdx in val_loader:
        print(gt.numpy(), ipIdx.numpy())
        
        stopAfter -= 1
        if stopAfter == 0:
            break
    
    # for ipIdx in train_loader:
    #     print(ipIdx)

    

    