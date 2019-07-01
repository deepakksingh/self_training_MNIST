'''contains a NN model for training MNIST'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN_Model(nn.Module):
    def __init__(self,num_of_input_channels,num_of_output_labels):
        super(MNIST_CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(num_of_input_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_of_output_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = MNIST_CNN_Model(num_of_input_channels = 1, num_of_output_labels = 10)
    x = torch.empty((100,1,28,28))
    output = model(x)
    print(output)

