 # 下面是resnet完整的代码
import os
import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report # 生成分类模型的详细性能评估报告

model = resnet18(num_classes=10)
model
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

class ResNetMNIST_L(L.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = resnet18(num_classes=10)
    self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.loss = nn.CrossEntropyLoss() # 确保模型和其他组件的属性被正确地存储在实例中

  def forward(self, x):
    assert isinstance(x, torch.cuda.FloatTensor), f"Input tensor should be torch.cuda.FloatTensor, but got {type(x)}"# 用 assert 语句来检查输入张量的类型
    return self.model(x)
  
  def training_step(self, batch, batch_no):
    x, y = batch # batch 通常是一个包含两个元素的 tuple，x 是输入数据，y 是对应的标签或目标值。
    x, y = x.cuda(), y.cuda()  # 将数据转移到 GPU
    logits = self(x) # 获取模型输出
    loss = self.loss(logits, y)
    return loss
  
  def configure_optimizers(self):
    return torch.optim.RMSprop(self.parameters(), lr=0.005) # 自适应学习率优化算法
  
model = ResNetMNIST_L().cuda()

# 创建训练器实例，管理训练的整个过程
trainer = L.Trainer(
    devices=1,
    max_epochs=1,
    log_every_n_steps=20  # 每 20 个步骤记录一次日志
)

trainer.fit(model, train_dl) # 启动训练流程

trainer.save_checkpoint("resnet18_mnist.pt")
  
def get_prediction(x, model: L.LightningModule): # PyTorch Lightning 框架的一部分
  model.freeze() # prepares model for predicting 
  probabilities = torch.softmax(model(x), dim=1)  #数据张量不一致
  predicted_class = torch.argmax(probabilities, dim=1)
  return predicted_class, probabilities

inference_model = ResNetMNIST_L.load_from_checkpoint("resnet18_mnist.pt", map_location="cuda")

true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch # 取数据和标签
  
  x = x.cuda() # 数据转移设备
  y = y.cuda() # 数据转移设备

  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())

# 假设 true_y 和 pred_y 可能是列表或张量
true_y = torch.tensor(true_y) if isinstance(true_y, list) else true_y
pred_y = torch.tensor(pred_y) if isinstance(pred_y, list) else pred_y

# 直接转换到 CPU 并转为 NumPy 数组
true_y_np = true_y.cpu().numpy() if true_y.device.type == 'cuda' else true_y.numpy()
pred_y_np = pred_y.cpu().numpy() if pred_y.device.type == 'cuda' else pred_y.numpy()

# 打印分类报告
print(classification_report(true_y_np, pred_y_np, digits=3))

