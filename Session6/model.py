import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) #o/p size=32*32*32 RF=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #o/p size=16*32*32 RF=5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
        ) #o/p size=16*16*16 RF=10

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=6*32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*32),
            nn.Conv2d(in_channels=6*32, out_channels=6*32, kernel_size=3, stride=1, padding=1, groups=6*32, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*32),
            nn.Conv2d(in_channels=6*32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #o/p size =32*16*16 RF=12

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
        ) #o/p size=32*8*8 RF=24
            
        # CONVOLUTION BLOCK 3       
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=6*32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*32),
            nn.Conv2d(in_channels=6*32, out_channels=6*32, kernel_size=3, stride=1, padding=1, groups=6*32, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*32),
            nn.Conv2d(in_channels=6*32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #o/p size = 64*8*8 RF = 26
        
        # TRANSITION BLOCK 3
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=0, bias=False),
        ) # #o/p size=64*4*4 RF=52

            
        # CONVOLUTION BLOCK 4       
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=6*64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*64),
            nn.Conv2d(in_channels=6*64, out_channels=6*64, kernel_size=3, stride=1, padding=1, groups=6*64, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*64),
            nn.Conv2d(in_channels=6*64, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 4 #o/p size = 128*4*4 RF = 52
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) #o/p size = 512*1*1 RF = 92

        self.linear = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)        
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x