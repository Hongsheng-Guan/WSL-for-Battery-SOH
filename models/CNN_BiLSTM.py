import torch
import torch.nn as nn
    
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(1,16,3),nn.ReLU(),nn.MaxPool1d(2))
        self.cnn2 = nn.Sequential(nn.Conv1d(16,32,3),nn.ReLU(),nn.MaxPool1d(2))

        self.lstm1_1 = nn.LSTM(32,16,batch_first=True,bidirectional=True)
        self.lstm1_2 = nn.Sequential(nn.Linear(32,32),nn.ReLU())
        self.lstm2_1 = nn.LSTM(32,16,batch_first=True,bidirectional=True)
        self.lstm2_2 = nn.Sequential(nn.Linear(32,32),nn.ReLU())

        self.flt = nn.Flatten(1,2)
        self.fc1 = nn.Linear(11*32,16)
        self.fc2 = nn.Linear(16,1)

    def forward(self, x):
        # x: [Batch, Input_length, Channel]
        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.permute(0,2,1)

        x,_ = self.lstm1_1(x)
        x = self.lstm1_2(x)
        x,_ = self.lstm2_1(x)
        x = self.lstm2_2(x)

        x = self.flt(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x