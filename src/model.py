import torch.nn.functional as F
import torch.nn as nn

class classify_conv_model(nn.Module):
    def __init__(self):
        super(classify_conv_model, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128,128)
        self.do1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 100)
        self.do2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(100, 2)
        
    def forward(self, x):
        # network performed well without batchnorm
        x = F.leaky_relu(self.conv1(x), inplace=True)
        x = F.leaky_relu(self.conv2(x), inplace=True)
        x = F.leaky_relu(self.conv3(x), inplace=True)
        x = x.view(-1,128)
        x = F.leaky_relu(self.fc1(x), inplace=True)
        x = self.do1(x)        
        x = F.leaky_relu(self.fc2(x), inplace=True)
        x = self.do2(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x