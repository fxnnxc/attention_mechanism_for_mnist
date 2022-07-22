

import torch 
from torch.utils.data import DataLoader, Dataset 
from torchvision.datasets import MNIST
import os 

class TokenDataset(Dataset):
    def __init__(self, train=True):
        if  not os.path.exists("untracked"):
            os.mkdir("untracked")
        mnist_data  = MNIST(root = "untracked/",  
                            train = train,  
                            download=True) 
        self.X = mnist_data.data
        self.Y = mnist_data.targets

    def __getitem__(self, idx):
        x_origin = self.X[idx]
        size = 4 # patch size
        stride = 4 # patch stride
        x = x_origin.unfold(0, size, stride).unfold(1, size, stride)
        x = x.flatten(start_dim=0, end_dim=1)  # patches 
        x = x.view(-1, size*stride) # 4x4 -> 16 vectorize
        return (x.float(), self.Y[idx], x_origin)

    def __len__(self):
        return len(self.X)
        


if __name__ == "__main__":
    dataset = TokenDataset()
    print(dataset[0][0].size())