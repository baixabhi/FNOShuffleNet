!pip uninstall torch torchvision torchaudio -y
!pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

import torch
print(torch.__version__)
print(torch.version.cuda)

!pip install neuraloperator==0.2.0
!pip install torch_harmonics==0.6.0

!pip show torch neuraloperator torch_harmonics
!pip install git+https://github.com/tensorly/torch
!pip install zarr
!pip install tensorly

from neuralop.models import TFNO

import numpy as np
import torch
import torchvision
from torchvision import models, datasets, transforms
import torch.nn as nncriterion
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
from neuralop.models import TFNO
from torchvision.transforms.functional import rotate, hflip, vflip, adjust_brightness, adjust_contrast
from collections import Counter
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformations for dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('/kaggle/input/msldv2/Train', transform=transform)
val_dataset = datasets.ImageFolder('/kaggle/input/msldv2/Valid', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Number of classes
classes = train_dataset.classes
num_classes = len(classes)
print("Classes:", classes)

# Define the attention and shuffle layers
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=8, reduction=4, n_modes=2, hidden_channels=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(n_modes)
        self.max_pool = nn.AdaptiveMaxPool2d(n_modes)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
          TFNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                in_channels=in_planes,
                out_channels=in_planes // reduction,
                factorization='tucker',
                implementation='factorized',
                rank=0.05),
          nn.ReLU(),
          TFNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                in_channels=in_planes,
                out_channels=in_planes // reduction,
                factorization='tucker',
                implementation='factorized',
                rank=0.05)

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(self.avg_pool(x).shape, self.max_pool(x).shape, self.fc(self.avg_pool(x)).shape)
        avg_out = self.avg_pool1(self.fc(self.avg_pool(x)))
        max_out = self.max_pool1(self.fc(self.max_pool(x)))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, n_modes=2, hidden_channels=16):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = TFNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                in_channels=2,
                out_channels=1,
                factorization='tucker',
                implementation='factorized',
                rank=0.05)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM_FNO_NeuralOp(nn.Module):
    def __init__(self, in_planes=8, hidden_channels=16,  n_modes=2, reduction=1, kernel_size=7):
        super(CBAM_FNO_NeuralOp, self).__init__()
        # Attention modules
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        #print(x.shape, self.channel_attention(x).shape)

        # Apply channel attention
        x = x * self.channel_attention(x)

        # Apply spatial attention
        x = x * self.spatial_attention(x)

        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        groups: Number of groups to divide the channels into.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0, "Number of channels must be divisible by groups."

        # Step 1: Reshape the input tensor to (batch_size, groups, num_channels // groups, height, width)
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # Step 2: Transpose to swap the groups and channels_per_group
        x = x.transpose(1, 2).contiguous()

        # Step 3: Flatten the tensor back to (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        return x

class FNOShuffle(nn.Module): #in_planes=8, hidden_channels=16,  n_modes=2, reduction=1, kernel_size=7
    def __init__(self, channels, n_modes=2, hidden_channels=16, reduction=1):
        super(FNOShuffle, self).__init__()
        self.split_size = channels // 2  # Split size (half the channels)
        self.cbam_fno = CBAM_FNO_NeuralOp(in_planes=self.split_size, reduction=reduction)
        self.channelshuffle = ChannelShuffle(groups=4)
        self.conv = TFNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                in_channels=self.split_size,
                out_channels=self.split_size,
                factorization='tucker',
                implementation='factorized',
                rank=0.05)

    def forward(self, x):
        # Split the input into two parts
        x1, x2 = torch.split(x, self.split_size, dim=1)
        x1 = self.cbam_fno(x1)
        x2 = self.conv(x2)
        x = torch.cat([x1,x2], dim=1)
        x = self.channelshuffle(x)

        return x

class FNOShuffleNet(nn.Module): #in_planes=8, hidden_channels=16,  n_modes=2, reduction=1, kernel_size=7
    def __init__(self, in_channels=16, num_classes=6, n_modes=4, hidden_channels=16, reduction=1):
        super(FNOShuffleNet, self).__init__()
        self.conv=nn.Conv2d(3,16,3,1,1)

        # Define the 8 layers using FNOShuffle and MaxPool
        self.layer1 = FNOShuffle(in_channels, n_modes=n_modes, hidden_channels=hidden_channels, reduction=reduction)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = FNOShuffle(in_channels, n_modes=n_modes, hidden_channels=hidden_channels, reduction=reduction)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = FNOShuffle(in_channels, n_modes=n_modes, hidden_channels=hidden_channels, reduction=reduction)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = FNOShuffle(in_channels, n_modes=n_modes, hidden_channels=hidden_channels, reduction=reduction)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5 = FNOShuffle(in_channels, n_modes=n_modes, hidden_channels=hidden_channels, reduction=reduction)
        self.pool5 = nn.MaxPool2d(kernel_size=16, stride=2)

        # Fully conch.Size([8nected classification layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, num_classes),  # Assuming input size after pooling is 8x8
        )

    def forward(self, x):
        # Pass through layers and MaxPool
        x = self.conv(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = self.pool5(self.layer5(x))

        # Classification
        x = self.fc(x)
        return x


model = FNOShuffleNet()
input_tensor = torch.randn(8, 3, 256, 256)
x = model(input_tensor)

# Training parameters
num_epochs = 40
learning_rate = 0.002
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Save the model
torch.save(model.state_dict(), 'fno_shuffle_net.pth')
print("Model saved successfully.")

# Load best model for evaluation
model.load_state_dict(torch.load('fno_shuffle_net.pth'))
model.eval()

# Evaluate on validation set
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# classification report and confusion matrix
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

