import torch
from torch import nn

class Reshape(nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=(2,2))
        self.Sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2),stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5))
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5))

        self.flatten = nn.Flatten()
        self.line = nn.Linear(120,84)
        self.output = nn.Linear(84,10)
    def forward(self,x):
        x = self.Sigmoid(self.conv1(x))
        x = self.avgpool(x)
        x = self.Sigmoid(self.conv2(x))
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.line(x)
        return self.output(x)
# if __name__ == '__main__':
    # lenet = LeNet()
    # x = torch.rand(size=(1,1,28,28),dtype=torch.float32)
    # print(lenet(x))

