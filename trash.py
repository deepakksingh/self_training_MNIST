import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == "__main__":
    
    #specify transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.5),(0.5))
        
        ]
    )

    initial_train_set = datasets.MNIST('./data', train = True, download = True, transform = transform)
    print(initial_train_set.__dict__.keys())
    train_split_percentage = 0.2
    initial_train_set_size = len(initial_train_set)
    
    needed_train_set_size = int(train_split_percentage * initial_train_set_size)
    
    needed_val_set_size = initial_train_set_size - needed_train_set_size

    print(needed_train_set_size, needed_val_set_size)
    train_set, val_set = torch.utils.data.random_split(initial_train_set, (needed_train_set_size, needed_val_set_size))
    # train_set, val_set = Dataset(train_set), Dataset(val_set)

    
    print((train_set.dataset))
    print(train_set.dataset.data.size())
    print(train_set.dataset.targets)
    
    print((val_set.dataset))
    print(val_set.dataset.data.size())
    print(val_set.dataset.targets)

    train_loader = torch.utils.data.DataLoader(val_set, batch_size = 100, shuffle = True, num_workers = 4)
    print(len(val_set))
