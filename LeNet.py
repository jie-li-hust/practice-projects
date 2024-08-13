import os
import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import lightning as L
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report # 生成分类模型的详细性能评估报告

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=3)  # 输入通道数为16，输出通道数为120，卷积核大小为3x3
        self.fc1 = nn.Linear(120 * 2 * 2, 84)  # 全连接层，输入特征数应与 conv3 的展平输出大小一致
        self.fc2 = nn.Linear(84, 10)  # 全连接层，输出特征数为10，对应10个类别

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))  # 卷积层 conv1 + 最大池化，输出形状：[batch_size, 6, 12, 12]
        
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))  # 卷积层 conv2 + 最大池化，输出形状：[batch_size, 16, 4, 4]
        
        x = F.relu(self.conv3(x))  # 卷积层 conv3，输出形状：[batch_size, 120, 2, 2]
        
        x = x.view(x.size(0), -1)  # 展平操作，输出形状：[batch_size, 120*2*2]
        
        x = F.relu(self.fc1(x))  # 全连接层 fc1
        
        x = self.fc2(x)  # 全连接层 fc2
        
        return x


train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)



# 定义 PyTorch Lightning 模块
class LeNetMNIST_L(L.LightningModule):
    def __init__(self):
        super(LeNetMNIST_L, self).__init__()
        self.model = LeNet()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x) # 获取输出
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=0.005)
  
# 实例化模型并将其转移到 GPU
model = LeNetMNIST_L().cuda()


# 训练并保存模型
trainer = L.Trainer(max_epochs=1)
model = LeNetMNIST_L().cuda()
trainer.fit(model, train_dl)
trainer.save_checkpoint("lenet_mnist.pt")


def get_prediction(x, model: L.LightningModule):
    model.freeze()  # 冻结模型参数以确保推理时不进行梯度更新
    model.eval()    # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算，节省内存，提高速度，不进行梯度计算
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1) # 计算
        predicted_class = torch.argmax(probabilities, dim=1) # 选择最大概率的类别：
    return predicted_class, probabilities


# 加载模型
inference_model = LeNetMNIST_L.load_from_checkpoint("lenet_mnist.pt", map_location="cuda")
inference_model.eval()  # 确保模型处于评估模式

true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch # 取数据和标签
  
  x = x.cuda() # 数据转移设备
  y = y.cuda() # 数据转移设备

  true_y.extend(y.cpu().numpy())  # 移动到 CPU 并转换为 NumPy 数组
  preds, _ = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu().numpy())  # 移动到 CPU 并转换为 NumPy 数组

# 将列表转换为 NumPy 数组
true_y_np = np.array(true_y)
pred_y_np = np.array(pred_y)


# 打印分类报告
print(classification_report(true_y_np, pred_y_np, digits=3))

