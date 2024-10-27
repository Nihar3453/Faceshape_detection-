# Face Shape Classification with PyTorch

This repository contains a PyTorch implementation of a face shape classification model using the EfficientNet architecture. The model is trained on a dataset of images, with data augmentation techniques applied to enhance performance.

## Features

- Utilizes EfficientNet for image classification.
- Implements data augmentation strategies for improved generalization.
- Supports learning rate scheduling to adapt to training progress.
- Saves the best model based on validation loss.

## Dataset

The model is trained on the [Face Shape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset). You can download it from Kaggle and place it in the appropriate directories.

## Requirements

To run this project, you will need the following packages:

- PyTorch
- torchvision
- scikit-learn
- Pillow
- other dependencies (listed in `requirements.txt`)
