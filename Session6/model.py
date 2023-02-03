import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) #o/p size=32*32*32 RF=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, groups=16, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) #o/p size=16*32*32 RF=5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
        ) # output_size = 32 #o/p size=16*32*32 RF=10

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=6*16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*16),
            nn.Conv2d(in_channels=6*16, out_channels=6*16, kernel_size=3, stride=1, padding=1, groups=6*16, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(6*16),
            nn.Conv2d(in_channels=6*16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 16  #o/p size =64*16*16 RF=10

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
        ) # output_size = 16 #o/p size=128*16*16 RF=14
            
        # CONVOLUTION BLOCK 3       
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3),groups=32, padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        #o/p size = 128*8*8 RF = 24
        
#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),            
#             nn.BatchNorm2d(128),
#             nn.Dropout(dropout_value)
#         ) # output_size = 8
        
        #o/p size = 256*8*8 RF = 32
        # TRANSITION BLOCK 3
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 8
        #o/p size=256*8*8 RF=32
        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 8
         #o/p size=256*4*4 RF=36
            
        # CONVOLUTION BLOCK 4       
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),groups=256, padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 4
        #o/p size = 256*4*4 RF = 52
        
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_value)
        ) # output_size = 4
        #o/p size = 512*4*4 RF = 68
                  
        
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1
        #o/p size = 512*1*1 RF = 92

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 
        #o/p size = 10*1*1 RF = 32


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        # x = self.pool1(x)
        x = self.convblock4(x)
#         x = self.convblock5(x)
        x = self.convblock6(x)
        # x = self.pool2(x)
        x = self.convblock7(x)
#         x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.pool3(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.gap(x)        
        x = self.convblock12(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)