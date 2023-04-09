# TSAI - EVA8 Session 13 Assignment

## Problem Statement

First part of your assignment is to train your own UNet from scratch, you can use the dataset and strategy provided in this [link](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406). However, you need to train it 4 times:
  
- MP+Tr+BCE  
- MP+Tr+Dice Loss  
- StrConv+Tr+BCE  
- StrConv+Ups+Dice Loss  

and report your results.  

Design a variation of a VAE that:

takes in two inputs:
an MNIST image, and
its label (one hot encoded vector sent through an embedding layer)
Training as you would train a VAE
Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!
Now do this for CIFAR10 and share 25 images (1 stacked image)!
Questions asked in the assignment are:
Share the MNIST notebook link ON GITHUB [100]
Share the CIFAR notebook link ON GITHUB [200]
Upload the 25 MNIST outputs PROPERLY labeled [250]
Upload the 25 CIFAR outputs PROPERLY labeled. [450]


