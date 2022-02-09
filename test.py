import torch
from LeNet import LeNet
from torch.autograd import Variable
from torchvision import datasets,transforms
from  torchvision.transforms import ToPILImage
from torch.utils import data

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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
model = LeNet()

model.load_state_dict(torch.load("save_model/best_model.pth",map_location='cpu'))

#获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

#把tensor转化为图片，方便可视化
show = ToPILImage()

#进入验证
for i in range(20):
    x,y = test_dataset[i][0],test_dataset[i][1]
    # print(x.shape,y)#torch.Size([1, 28, 28]) 7
    show(x).show()
    x = Variable(torch.unsqueeze(x,dim=0).float(),requires_grad = False)#[1, 28, 28]-->[1, 1, 28, 28]
    #unqueeze 去掉维度
    #queeze 增加维度6

    #将tensor(张量)转化成variable(变量)。之所以需要将tensor转化成variable是因为pytorch中tensor(张量)只能放在CPU上运算，
    # 而(variable)变量是可以只用GPU进行加速计算的
    with torch.no_grad():
        pred = model(x)
        # print(pred.shape)#torch.Size([1, 10])
        predicted,actual = classes[torch.argmax(pred[0])],classes[y]
        print(f"最后的结果:{predicted},actual = {actual}")
