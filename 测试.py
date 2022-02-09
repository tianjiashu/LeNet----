import torch
from torch import nn
from torch.utils import data
from LeNet import LeNet
from torch.optim import lr_scheduler
from torchvision import datasets,transforms
import os

batch_size = 16

train_dataset = datasets.MNIST(
    root="./dataset",train=True,transform=transforms.ToTensor(),download=True
)
test_dataset = datasets.MNIST(
    root="./dataset",train=False,transform=transforms.ToTensor(),download=True
)

train_loader = data.DataLoader(
    dataset = train_dataset,batch_size=batch_size,shuffle=True
)
test_loader = data.DataLoader(
    dataset = test_dataset,batch_size=batch_size,shuffle=True
)

# print(f"test_loader包含{len(test_loader)}个批次")#test_loader包含625个批次
# for batch,(x,y) in enumerate(test_loader):
#     print(batch,x.shape,y)#481 torch.Size([16, 1, 28, 28]) tensor([5, 2, 9, 7, 4, 4, 3, 5, 6, 9, 1, 9, 8, 7, 4, 6])
# for test in test_loader:
#     print(test[0].shape,test[1])#torch.Size([16, 1, 28, 28]) tensor([1, 1, 3, 0, 7, 9, 3, 7, 2, 0, 7, 8, 9, 5, 2, 6])