import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2,stride=2),nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64,kernel_size=2,stride=2),nn.ReLU())
        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128,kernel_size=4,stride=1),nn.ReLU())
        self.flt = nn.Flatten(1,2)
        self.fc1 = nn.Sequential(nn.Linear(128*9,128*9),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128*9,1))

    def forward(self, x):
        # x: [Batch, Input_length, Channel]
        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flt(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x