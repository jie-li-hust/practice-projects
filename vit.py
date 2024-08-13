import argparse
import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from transformer_package.models import ViT

# 解析命令行参数
parser = argparse.ArgumentParser(description="Train or test the ViT model on MNIST dataset.")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Run mode, either train or test.')
args = parser.parse_args()

# 模型参数
image_size = 28
channel_size = 1
patch_size = 7
embed_size = 128
num_heads = 4
classes = 10
num_layers = 2
hidden_size = 128
dropout = 0.1

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTModel(pl.LightningModule):
    def __init__(self, image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout):
        super(ViTModel, self).__init__()
        self.model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        loss = self.criterion(predictions, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        loss = self.criterion(predictions, labels)
        acc = (predictions.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        loss = self.criterion(predictions, labels)
        acc = (predictions.argmax(dim=1) == labels).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        # 配置优化器和学习率调度器
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

data_path = "/home/jieli/pytorch-resnet-mnist/mnist/MNIST"
train_dataset = MNIST(root=data_path, train=True, transform=transform, download=True)
test_dataset = MNIST(root=data_path, train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 根据模式决定流程
if args.mode == 'train':
    # 实例化模型
    model = ViTModel(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout)

    # 初始化Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        devices=1,
        accelerator='gpu'
    )

    # 训练模型
    trainer.fit(model, train_loader, val_dataloaders=test_loader)

    # 保存模型
    trainer.save_checkpoint("vit_mnist_final.pt")
elif args.mode == 'test':
    # 加载模型
    model = ViTModel(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout)
    checkpoint = torch.load("vit_mnist_final.pt")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    model.eval()

    # 初始化Trainer
    trainer = pl.Trainer(devices=1, accelerator='gpu')

    # 测试模型
    results = trainer.test(model, dataloaders=test_loader)
    print(results)
else:
    raise ValueError("Invalid mode specified. Please choose 'train' or 'test'.")

# 训练命令和测试命令如下：
# 训练模式：python vit.py --mode=train
# 测试模式：python vit.py --mode=test