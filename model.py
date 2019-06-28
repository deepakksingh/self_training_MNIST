'''contains a NN model for training MNIST'''
import torch
import torch.nn as nn

class MNIST_Model(nn.Module):
    '''define a custom NN for MNIST'''
    def __init__(self, num_of_input_channels, num_of_output_channels):
        '''initialize the model with given channels'''
        super(MNIST_Model, self).__init__()
        self.num_of_input_channels = num_of_input_channels
        self.num_of_output_channels = num_of_output_channels

        self.create_nn()

    def create_nn(self):
        self.l1 = nn.Linear(in_features = self.num_of_input_channels, out_features = 10, bias = True)
        self.l2 = nn.Linear(in_features = 10, out_features = 10, bias = True)
        self.l3 = nn.Linear(in_features = 10, out_features = 20, bias = True)
        self.l4 = nn.Linear(in_features = 20, out_features = 20, bias = True)
        self.l5 = nn.Linear(in_features = 20, out_features = self.num_of_output_channels, bias = True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        '''define forward propagation'''
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.relu(self.l5(x))
        # x = self.softmax(x)
        
        return x

if __name__ == "__main__":
    model = MNIST_Model(784, 10)
    x = torch.randn(784)
    output = model(x)
    print(output)

