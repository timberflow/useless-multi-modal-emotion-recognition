import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, input_channel, channels):
        
        super().__init__()

        self.p1 = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], (1, 1)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(input_channel, channels[1][0], (1, 1)),
            nn.BatchNorm2d(channels[1][0]),
            nn.ReLU(),
            nn.Conv2d(channels[1][0], channels[1][1], (3, 3), padding = 1),
            nn.BatchNorm2d(channels[1][1]),
            nn.ReLU()
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(input_channel, channels[2][0], (1, 1)),
            nn.BatchNorm2d(channels[2][0]),
            nn.ReLU(),
            nn.Conv2d(channels[2][0], channels[2][1], (5, 5), padding = 2),
            nn.BatchNorm2d(channels[2][1]),
            nn.ReLU()
        )

        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(input_channel, channels[3], (1, 1)),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
        )


    def forward(self, x):
        y1 = self.p1(x)
        y2 = self.p2(x)
        y3 = self.p3(x)
        y4 = self.p4(x)
        return torch.cat((y1, y2, y3, y4), dim = 1)
    
class GoogLeNet(nn.Module):
    def __init__(self, input_channel, input_size, output_size):
        
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, (7, 7), stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride = 2, padding = 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, (3, 3), padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride = 2, padding = 1)
        )
        
        self.inception3a = Inception(192, (64, (96, 128), (16, 32), 32))
        self.inception3b = Inception(256, (128, (128, 192), (32, 96), 64))
        
        self.inception4a = Inception(480, (192, (96, 208), (16, 48), 64))
        self.inception4b = Inception(512, (160, (112, 224), (24, 64), 64))
        self.inception4c = Inception(512, (128, (128, 256), (24, 64), 64))
        self.inception4d = Inception(512, (112, (144, 288), (32, 64), 64))
        self.inception4e = Inception(528, (256, (160, 320), (32, 128), 128))
        
        self.inception5a = Inception(832, (256, (160, 320), (32, 128), 128))
        self.inception5b = Inception(832, (384, (192, 384), (48, 128), 128))

        self.maxpool = nn.MaxPool2d((3, 3), stride = 2, padding = 1)
        self.avgpool = nn.AvgPool2d(input_size // 32, 1)
        
    
    
    def forward(self, x):
        L = self.conv2(self.conv1(x))
        L = self.inception3b(self.inception3a(L))
        L = self.maxpool(L)
        L = self.inception4e(self.inception4d(self.inception4c(self.inception4b(self.inception4a(L)))))
        L = self.maxpool(L)
        L = self.inception5b(self.inception5a(L))
        
        return L