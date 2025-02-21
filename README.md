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


## Acknowledgments
Built using **PyTorch** and inspired by **Fourier Neural Operators (FNO)** for efficient learning.
