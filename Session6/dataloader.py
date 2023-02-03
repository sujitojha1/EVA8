import torch
import torchvision
import torchvision.transforms as transforms


# Defining CUDA
cuda = torch.cuda.is_available()

print("CUDA availability ?",cuda)


# Transformations in training phase
train_transform = transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                      ])

# Transformations in testing phase
test_transform = transforms.Compose([
#                                      transforms.RandomCrop(32, padding=4),
#                                      transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])

# Loading train data
train = datasets.CIFAR10('./Data',train=True,transform=train_transform,download=True)

# Loading test data
test = datasets.CIFAR10('./Data',train=False, transform=test_transform,download=True)

# Defining data loaders with setting
dataloaders_args = dict(shuffle=True, batch_size=128, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True,batch_size=64)

# Train dataloader
trainloader = torch.utils.data.DataLoader(train, **dataloaders_args)

# Test dataloader
testloader = torch.utils.data.DataLoader(test, **dataloaders_args)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')