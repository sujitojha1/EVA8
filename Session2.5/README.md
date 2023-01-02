# TSAI - EVA8 Session 2.5 Assignment

## Problem Statement

**Write a neural network that can:**
1. take 2 inputs:  
    1. an image from the MNIST dataset (say 5), and  
    2. a random number between 0 and 9, (say 7)
2. and gives two outputs:  
    1. the "number" that was represented by the MNIST image (predict 5), and  
    2. the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)
    <img src="https://user-images.githubusercontent.com/30425824/138210916-3ab38f38-6508-4a48-a4aa-5c5ef31d9e0f.png" width="500"/>

3. you can mix fully connected layers and convolution layers  
4. you can use one-hot encoding to represent the random number input as well as the "summed" output.  
    a. Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0  
    b. Sum (13) can be represented as: 
        1. 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  
        2. 0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010

**Your code MUST be:**
1. well documented (via readme file on github and comments in the code)
2. must mention the data representation
3. must mention your data generation strategy (basically the class/method you are using for random number generation)
4. must mention how you have combined the two inputs (basically which layer you are combining)
5. must mention how you are evaluating your results 
6. must mention "what" results you finally got and how did you evaluate your results
7. must mention what loss function you picked and why!  
8. training MUST happen on the GPU
9. Accuracy is not really important for the SUM


**Once done, upload the code with short training logs in the readme file from colab to GitHub, and share the GitHub link (public repository)**

## Solution & Discussion
- **Two notebooks (trained using GPU) with results**

    | Methods        | Notebook      | Hyperparameters: Optimizer,Learning_rate  |  Results: Accuracies  |
    |:----------------------------: |:---------------------:|:---------------------:| :--------|
    | Convolutional Layers with Fully connected Layer | [Convolutional Layers + FC](Session3_Pytorch101_ver3.ipynb) |Optimizer=SGD, Learning_rate = 0.1, epochs = 20, batch_size = 100|  Image  = 99% and Sum Label  = 99% |

- **Two methods were tried in this study (with details of how the two inputs are combined)**

    a. Method1: Neural Network with **Convolutional Layers & Fully Connected Layers**  

    <img src="https://user-images.githubusercontent.com/30425824/137436572-8f274f50-0b73-4dcf-87b4-42d1bd56494d.png" alt="Method1" width="800"/>


- **Data Representation**   

    <img src="https://user-images.githubusercontent.com/30425824/137435089-b231d73c-ee7e-406e-82d1-de975b62caa6.png" alt="DataRepresentation" width="800"/>
    <img src="https://user-images.githubusercontent.com/30425824/137435132-41627581-b165-4165-a108-a951ff964244.png" alt="DataRepresentation" width="800"/>

 
- **Data Generation Strategy**  
    Leveraged torchvision mnist dataset and defined the custom data class to generate the random number with its one hot encoding.
    ```Python
    # Loading the mnist dataset from pytorch - torchvision
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    mnist_train = datasets.MNIST('../data',train=True,download=True) # Train dataset
    mnist_test = datasets.MNIST('./data',train=False,download=True) # Test dataset

    # Custom Data Class
    from torch.utils.data import Dataset
    from random import randrange

    # Dataset is there to be able to interact with DataLoader

    class MyDataset(Dataset):
    def __init__(self, inpDataset, transform):
        self.inpDataset = inpDataset
        self.transform = transform

    def __getitem__(self, index):
        randomNumber = randrange(10)
        sample_image, label = self.inpDataset[index]
        if self.transform:
            sample_image = self.transform(sample_image)

        sample = (sample_image,F.one_hot(torch.tensor(randomNumber),num_classes=10), 
                label,label+randomNumber)
        return sample

    def __len__(self):
        return len(self.inpDataset)

    myData_train = MyDataset(mnist_train,transform) 
    myData_test = MyDataset(mnist_test,transform)

    # One Sample output
    # (torch.Size([1, 28, 28]), tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 5, 14)
    ```

- **Results Evaluation Strategy**  
    Used accuracies as the metric to evaluate the performance of the models. Accuracy = (total images or sum label predicted correctly) / (total training/test length)

- **Loss functions**  
    - ```Negative Log Likelihood Loss``` is used in the model since model is predicting a multiclass output. We calculate loss separately for each output:  
        - Loss1 --> Loss between actual image label and predicted image label
        - Loss2 --> Loss between actual sum label and predicted sum label
        - Total Loss = Loss1 + Loss2  

    - ```Cross Entropy Loss``` could also be used but we need to use ```softmax``` in the prediction layer.
    - ```Binary Cross Entropy Loss``` couldnt be used here since we have more than 2 labels in each of the outputs.  
    - ```Mean Squared Loss``` If used, it would converge in large number of epochs as it would penalise more the  deviation from the ground truth.


## Reference
1. https://github.com/pytorch/examples/blob/master/mnist/main.py
2. https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
