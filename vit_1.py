import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class SimpleViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, embed_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super(SimpleViT, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim # 嵌入维度
        self.num_heads = num_heads # 注意力头的数量
        self.num_layers = num_layers # Transformer 层的数量
        self.num_classes = num_classes
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Linear(patch_size * patch_size * 1, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim*2
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * 1)
        x = self.patch_embed(x)
        x = x + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

class ViTLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(ViTLightning, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
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
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    train_ds = MNIST("mnist", train=True, download=True, transform=transform)
    test_ds = MNIST("mnist", train=False, download=True, transform=transform)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    return train_dl, test_dl

if __name__ == "__main__":
    model = SimpleViT()
    lit_model = ViTLightning(model)

    train_dl, test_dl = load_mnist_data()

    trainer = pl.Trainer(
        max_epochs=5,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    trainer.fit(lit_model, train_dl, test_dl)
