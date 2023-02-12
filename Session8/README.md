# TSAI - EVA8 Session 8 Assignment

## Problem Statement

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:  
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k] 
    2. Layer1 -  
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]  
        2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        3. Add(X, R1)  
    3. Layer 2 -  
        1. Conv 3x3 [256k]  
        2. MaxPooling2D  
        3. BN  
        4. ReLU  
    4. Layer 3 -  
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]  
        2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]  
        3. Add(X, R2)   
    5. MaxPooling with Kernel Size 4    
    6. FC Layer  
    7. SoftMax  
2. Uses One Cycle Policy such that:  
    1. Total Epochs = 24  
    2. Max at Epoch = 5  
    3. LRMIN = FIND  
    4. LRMAX = FIND  
    5. NO Annihilation  
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)  
4. Batch size = 512  
5. Target Accuracy: 90% (93.8% quadruple scores).  
6. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training.  
7. Once done, proceed to answer the Assignment-Solution page. 

## Solution

This readme file provides a complete documentation for the custom ResNet architecture for CIFAR10 dataset. The custom ResNet architecture has been designed to meet the requirements specified in the problem statement.

## Architecture

The custom ResNet architecture is trained using One Cycle Policy with the following parameters:

Total epochs: 24
Maximum at epoch: 5
LRMIN: To be determined during training
LRMAX: To be determined during training
No Annihilation
The training data is transformed using the following operations:

RandomCrop (32, 32) after padding of 4
FlipLR
CutOut (8, 8)
Batch size: 512
The target accuracy for the custom ResNet architecture is 90% (93.8% quadruple scores).

## Modularity
The code for the custom ResNet architecture is modular and is available in the custom_resnet.py file in the GitHub repository. The collab is set up to import the custom_resnet package from GitHub and run the model.

## Conclusion
The custom ResNet architecture for CIFAR10 dataset has been designed and documented to meet the requirements specified in the problem statement. The code for the custom ResNet architecture is modular and is available in the GitHub repository. The target accuracy for the custom ResNet architecture is 90% (93.8% quadruple scores).
