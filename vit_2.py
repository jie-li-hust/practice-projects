import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# 定义 PatchEmbedding 模块
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类标记
        self.pos_emb = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))  # 位置编码

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # 将卷积操作生成的补丁特征转换成Transformer所需的格式
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = x + self.pos_emb
        return x

# 定义 MultiHeadAttention 模块
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.transpose(0, 1)  # 转置为 (seq_len, batch, embed_dim) 形式
        attn_output, _ = self.attention(x, x, x, need_weights=False)  # 获取注意力输出
        x = attn_output.transpose(0, 1)  # 转回 (batch, seq_len, embed_dim)
        return x

# 定义 MLP 模块
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

# 定义 Transformer Encoder Layer 模块
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 定义 ViT 模型
class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=128, depth=6, num_heads=4, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.head(x[:, 0])  # 取分类标记的输出
        return x

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 实例化模型
model = ViT(img_size=28, patch_size=7, in_channels=1, embed_dim=128, depth=6, num_heads=4, num_classes=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# 测试模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# 训练并测试模型
train_model(model, train_loader, criterion, optimizer, num_epochs=5)
evaluate_model(model, test_loader)
