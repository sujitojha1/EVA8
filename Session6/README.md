# TSAI - EVA8 Session 6 Assignment

## Problem Statement

1. Run this [network](https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw)  
2. Fix the network above:  
    1. change the code such that it uses GPU and  
    2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)   
    3. total RF must be more than 44  
    4. one of the layers must use Depthwise Separable Convolution  
    5. one of the layers must use Dilated Convolution
    use GAP (compulsory)  
    6. add FC after GAP to target #of classes (optional)   
    7. use albumentation library and apply:  
        1. horizontal flip
        2. shiftScaleRotate
        3. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)  
    8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.  
    9. upload to Github
    10. Attempt S6-Assignment Solution.  
    11. Questions in the Assignment QnA are:  
        1. copy paste your model code from your model.py file (full code)  
        2. copy paste output of torchsummary 
        3. copy-paste the code where you implemented albumentation transformation for all three transformations  
        4. copy paste your training log (you must be running validation/text after each Epoch  
        5. Share the link for your README.md file. 

## Introduction

The goal of this assignment is to design a Convolutional Neural Network (CNN) using PyTorch and the Albumentation library to achieve an accuracy of 85% on the CIFAR10 dataset. The code for this assignment is provided in a Jupyter Notebook, which can be found [here](./EVA8_S6_CIFAR10.ipynb).

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The 10 classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 validation images.

## Model Architecture

The model for this project is based on the C1C2C3C40 architecture with several modifications. Instead of max pooling, the network consists of 3 convolutional layers with 3x3 filters and a stride of 2. The final layer utilizes global average pooling (GAP). One layer uses depthwise separable convolution, while another layer uses dilated convolution. The architecture leverages mobileNetV2, which combines expand, depthwise, and pointwise convolution with residual connections.
Data Augmentation

## Data augmentation 
Augmentation is performed using the Albumentations library. Three techniques are applied in the training data loader: horizontal flipping, shiftScaleRotate, and coarseDropout. No dropout was included in the model as these data augmentation methods provide similar regularization effects.

Sample images,  
![augmentation](./images/dataloader_preview.png)

## Results

The model was trained for 25 epochs and achieved an accuracy of 84.64% on the test set. The total number of parameters in the model was under 200k. The training logs, as well as the output of torchsummary, are included in this notebook.

Trainling accuracy: 82.84 %
Test accuracy: 84.64 %

## Classwise Accuracy

![classwise_accuracy](./images/classwise_accuracy.png)

## Misclassified Images

Few Samples of misclassified images,  
![misclassified](./images/misclassified_images.png)
