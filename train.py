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
#test_loader 包含 625个批次，每个批次包含16张图片
#for batch,(x,y) in enumerate(test_loader)  batch 批次，x 16张图片，y 16张图片的标签。
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet().to(device)

loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
#学习率每隔10轮变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optim,step_size=10,gamma=0.1)

def train(dataloader,model,loss,optim):
    loss_sum,current,n = 0.0,0.0,0
    for batch,(x,y) in enumerate(dataloader):#x是图片，y是标签
        # print(x.shape,y)#torch.Size([16, 1, 28, 28]) tensor([9, 3, 6, 3, 1, 7, 6, 4, 8, 0, 7, 1, 9, 3, 5, 1])
        X,Y = x.to(device),y.to(device)
        output = model(X)
        cur_loss = loss(output,Y)
        # print(output.shape)#torch.Size([16, 10])
        ans,pred = torch.max(output,axis = 1)#axis = 1取每一行的最大值
        #ans = tensor([0.1622, 0.1617, 0.1609, 0.1602, 0.1624, 0.1606, 0.1609, 0.1609, 0.1607,0.1597, 0.1600, 0.1611, 0.1602, 0.1596, 0.1596, 0.1617],grad_fn=<MaxBackward0>)
        #pred = tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        #一张图片返回好几个结果，找最大结果，pred标签
        cur_acc = torch.sum(Y==pred)/output.shape[0]
        #output.shape[0] = 16 16张图片
        optim.zero_grad()
        cur_loss.backward()
        optim.step()

        loss_sum+=cur_loss.item()
        current +=cur_acc.item()
        n+=1
    print(f"训练的loss值{loss_sum/n},精确度为{current/n}")

def val(dataloader,model,loss):
    model.eval()
    loss_sum,current,n = 0.0,0.0,0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):  # x是图片，y是标签
            X, Y = x.to(device), y.to(device)
            output = model(X)
            cur_loss = loss(output, Y)
            _, pred = torch.max(output, axis=1)  # axis = 1取每一行的最大值
            cur_acc = torch.sum(Y == pred) / output.shape[0]

            loss_sum += cur_loss.item()
            current += cur_acc.item()
            n += 1
        print(f"验证的loss值{loss_sum / n},精确度为{current / n}")
        return current/n


epoch = 50
min_acc = 0

for i in range(epoch):
    print(f"第{i+1}轮训练","------"*10)
    train(train_loader,model,loss,optim)
    a = val(test_loader,model,loss)
    if a>min_acc:
        folder = "save_model"
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = a
        print("保存最好的模型")
        torch.save(model.state_dict(),"save_model/best_model.pth")
print('Done!')

