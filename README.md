# FNOShuffleNet: Monkeypox detection using Attention and Channel Shuffle

## Overview
This project implements **FNOShuffleNet**, a neural network for image classification. The model integrates **Fourier Neural Operators (FNO)** with **Channel Attention, Spatial Attention, and Channel Shuffle** mechanisms to enhance feature extraction.

## Features
- Uses **CBAM (Convolutional Block Attention Module)** for improved attention.
- **Channel Shuffle** enhances information flow between feature maps.
- Implements **FNO-based convolutions** for efficient learning.
- Trained using **PyTorch** with data augmentation.

## Dataset
- The model is trained on images from the **MSLDV2 dataset**.
- Data is preprocessed using transformations like resizing, flipping, rotation, and color jittering.

## Model Architecture
1. **FNOShuffle Layers** for feature extraction.
2. **MaxPooling Layers** for downsampling.
3. **Fully Connected Layer** for classification.

## Installation
```bash
pip install torch torchvision
```

## Usage
```python
import torch
from model import FNOShuffleNet

# Load model
model = FNOShuffleNet(num_classes=6)

# Dummy input
input_tensor = torch.randn(8, 3, 256, 256)
output = model(input_tensor)
print(output.shape)  # Output: (8, 6)
```

## Training
Modify the dataset path and run:
```python
python train.py
```

## Acknowledgments
Built using **PyTorch** and inspired by **Fourier Neural Operators (FNO)** for efficient learning.
