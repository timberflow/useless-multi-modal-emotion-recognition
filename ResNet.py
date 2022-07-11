import torch
import torch.nn as nn
import torch.nn.functional as F

class ResCell(nn.Module):
    def __init__(self, channel, kernel_size, padding = 0):
        
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding = padding),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding = padding),
            nn.BatchNorm2d(channel)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        score = self.relu(x + self.conv(x))
        return score
        
class ResNet(nn.Module):
    def __init__(self, input_channel, input_size, output_size, alignment_size = 768):
        
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.rescell_1 = nn.Sequential(
            ResCell(64, 3, 1),
            ResCell(64, 3, 1),
            ResCell(64, 3, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.rescell_2 = nn.Sequential(
            ResCell(128, 3, 1),
            ResCell(128, 3, 1),
            ResCell(128, 3, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.rescell_3 = nn.Sequential(
            ResCell(256, 3, 1),
            ResCell(256, 3, 1),
            ResCell(256, 3, 1),
            ResCell(256, 3, 1),
            ResCell(256, 3, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, alignment_size, 3, 2, 1),
            nn.BatchNorm2d(alignment_size),
            nn.ReLU(),
            nn.Conv2d(alignment_size, alignment_size, 3, 1, 1),
            nn.BatchNorm2d(alignment_size),
            nn.ReLU()
        )
        self.rescell_4 = nn.Sequential(
            ResCell(alignment_size, 3, 1),
            ResCell(alignment_size, 3, 1)
        )
        
        self.avgpool = nn.AvgPool2d(input_size // 32, 1)
        self.alignment = nn.Sequential(
            nn.Linear(alignment_size, alignment_size),
            nn.ReLU()
        )
        
    
    def forward(self, x, unpool = False):
        L = self.rescell_1(self.conv1(x))
        L = self.rescell_2(self.conv2(L))
        L = self.rescell_3(self.conv3(L))
        L = self.rescell_4(self.conv4(L))
        if unpool:
            L = L.contiguous().view(L.shape[0], L.shape[1], -1)
            L = L.contiguous().permute([0,2,1])
        else:
            L = self.avgpool(L)
            L = L.contiguous().view(L.shape[0], -1)
            #L = self.alignment(L)
        
        return L