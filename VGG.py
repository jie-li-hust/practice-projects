import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchvision.models import vgg16
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, Lambda
from torch.utils.data import DataLoader
import lightning as L
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report

# 数据预处理
def to_rgb(image):
    return image.convert("RGB")  # 将灰度图像转换为伪 RGB 图像

transform = Compose([
    Lambda(to_rgb),  # 转换为 RGB 图像
    Resize(224),  # 调整图片大小以适应 VGG 输入
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG 的标准化值
])

train_ds = MNIST("mnist", train=True, download=True, transform=transform)
test_ds = MNIST("mnist", train=False, download=True, transform=transform)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

class VGG_MNIST_L(L.LightningModule):
    def __init__(self):
        super(VGG_MNIST_L, self).__init__()
        self.model = vgg16(pretrained=True)  # 使用预训练的 VGG16 模型
        self.model.classifier[6] = nn.Linear(4096, 10)  # 修改全连接层，适配 MNIST 数据集（10 类）

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)  # 获取输出
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=0.005)

# 实例化模型并将其转移到 GPU
model = VGG_MNIST_L().cuda()

# 训练并保存模型
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dl)
trainer.save_checkpoint("vgg_mnist.pt")

def get_prediction(x, model: L.LightningModule):
    model.freeze()  # 冻结模型参数以确保推理时不进行梯度更新
    model.eval()    # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)  # 计算预测概率
        predicted_class = torch.argmax(probabilities, dim=1)  # 选择最大概率的类别
    return predicted_class, probabilities

# 加载模型
inference_model = VGG_MNIST_L.load_from_checkpoint("vgg_mnist.pt", map_location="cuda")
inference_model.eval()  # 确保模型处于评估模式

true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
    x, y = batch  # 取数据和标签

    x = x.cuda()  # 数据转移到 GPU
    y = y.cuda()  # 数据转移到 GPU

    true_y.extend(y.cpu().numpy())  # 移动到 CPU 并转换为 NumPy 数组
    preds, _ = get_prediction(x, inference_model)
    pred_y.extend(preds.cpu().numpy())  # 移动到 CPU 并转换为 NumPy 数组

# 将列表转换为 NumPy 数组
true_y_np = np.array(true_y)
pred_y_np = np.array(pred_y)

# 打印分类报告
print(classification_report(true_y_np, pred_y_np, digits=3))
