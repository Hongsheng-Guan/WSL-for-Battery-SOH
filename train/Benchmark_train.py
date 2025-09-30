import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.CNN_BiLSTM import CNN_BiLSTM
import torch.optim as optim
import numpy as np
from tool.metrix import eval_metrix
import os

class MyDataset(Dataset):
    def __init__(self,X,Y):
        super(MyDataset,self).__init__()
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()
        self.X = X.view(-1,50,1)
        self.Y = Y.view(-1,1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        xi,yi = self.X[index],self.Y[index]
        return xi,yi
    
class BenchmarkTrain():
    def __init__(self,args,batch_size=4) -> None:
        self.target_dataset = args.target_dataset
        self.batch_size = batch_size
        self.epochs = 500
        self.lr = 0.001

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ft_files = args.ft_files
        self.loss_func = nn.MSELoss()

    def train(self,X,Y):
        """
        Train the model
        训练模型
        """
        print('Training Benchmark model')
        data = MyDataset(X,Y)
        model = CNN_BiLSTM().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        train_loss = []
        
        for epoch in range(self.epochs):
            # Split the dataset into training and validation sets
            # 将数据集分为训练集和验证集
            train_loader = DataLoader(data,batch_size=self.batch_size,shuffle=True)
            epoch_train_loss = []

            # Training loop
            # 训练循环
            for x,y in train_loader:
                x,y = x.to(self.device),y.to(self.device)
                y_hat = model(x)
                loss = self.loss_func(y,y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))
            print(f'{self.target_dataset} train_epoch: {epoch} --- train_loss: {train_loss[-1]}')
            model_path = os.path.join(self.ft_files, 'Benchmark_model.pth')
            torch.save(model.state_dict(), model_path)

        # Save the loss
        # 保存损失
        loss_path = os.path.join(self.ft_files, 'Benchmark_loss.npz')
        np.savez(loss_path, train_loss=train_loss)

    def test(self,X,true_c):
        """
        Test the model
        测试模型
        """
        model = CNN_BiLSTM().to(self.device)
        model_path = os.path.join(self.ft_files, f'Benchmark_model.pth')
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        est_capacity = []
        for xi in X:
            xi = torch.tensor(xi, dtype=torch.float).to(self.device)
            xi = xi.view(1, 50, 1)
            est_y = model(xi)
            est_y = est_y[0, 0]
            est_capacity.append(est_y.cpu().detach().numpy())

        # est_capacity = np.array(est_capacity)*C0 
        metrics = eval_metrix(true_c,est_capacity)
        return est_capacity,metrics