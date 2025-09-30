import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.SSL import Autoencoder
from tool.metrix import eval_metrix
import torch.optim as optim
import numpy as np
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

class SSLTrain(): 
    def __init__(self,args) -> None:
        """
        Initialize the SSLTrain class, used to train the autoencoder
        初始化 SSLTrain 类，用于预训练自编码器
        """
        self.source_dataset = args.source_dataset
        self.pre_epochs = 300
        self.pre_batch_size = 1024
        self.pre_lr = 0.001    

        self.target_dataset = args.target_dataset
        self.ft_epochs = 500
        self.ft_batch_size = 4
        self.ft_lr = 0.0005

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_folder = args.save_folder 
        self.ft_files = args.ft_files
        self.loss_func = nn.MSELoss()

    def pre_train(self,X,Y):
        """
        pre-train the model
        预训练模型
        """
        train_size = int(len(X)*0.7)
        val_size = len(X) - train_size
        data = MyDataset(X,Y)
        model = Autoencoder().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.pre_lr)
        train_loss,val_loss = [],[]
        val_loss_crit = None

        for epoch in range(self.pre_epochs):
            # Split the dataset into training and validation sets
            # 将数据集分为训练集和验证集
            train_dataset, val_dataset = torch.utils.data.random_split(data,[train_size,val_size])
            train_loader = DataLoader(train_dataset,batch_size=self.pre_batch_size,shuffle=True)
            epoch_train_loss = []

            # Training loop
            # 训练循环
            for x,_ in train_loader:
                x = x.view(-1,50).to(self.device)
                x_hat,_ = model(x)
                loss = self.loss_func(x,x_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))

            # Validation loop
            # 验证循环
            with torch.no_grad():
                val_loader = DataLoader(val_dataset,batch_size=self.pre_batch_size,shuffle=True)
                epoch_val_loss = []

                for x,_ in val_loader:
                    x = x.view(-1,50).to(self.device)
                    x_hat,_ = model(x)
                    loss = self.loss_func(x,x_hat)
                    epoch_val_loss.append(loss.item())
                val_loss.append(np.mean(epoch_val_loss))

                # Save the model with the lowest validation loss
                # 保存验证损失最低的模型
                model_path = os.path.join(self.save_folder, 'SSL_pre_trained_model.pth')
                if val_loss_crit is None or val_loss_crit >= val_loss[-1]:
                    val_loss_crit = val_loss[-1]
                    torch.save(model.state_dict(), model_path)

            print(f'{self.source_dataset} pre_train_epoch: {epoch} --- train_loss: {train_loss[-1]} --- val_loss: {val_loss[-1]} --- val_loss_crit: {val_loss_crit}')
        loss_path = os.path.join(self.save_folder, 'SSL_pre_trained_loss.npz')
        np.savez(loss_path, train_loss=train_loss, val_loss=val_loss)
    
    def ft_train(self,X,Y):
        """
        fine-tune the model
        微调模型
        """
        data = MyDataset(X,Y)
        model = Autoencoder().to(self.device)
        pre_model_path = os.path.join(self.save_folder, 'SSL_pre_trained_model.pth')
        model.load_state_dict(torch.load(pre_model_path,weights_only=True))

        # Freeze the encode layers
        # 冻结encoder层
        for name, parameter in model.named_parameters():
            if 'regressor' in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

        optimizer = optim.Adam(model.parameters(),lr=self.ft_lr)
        train_loss = []

        for epoch in range(self.ft_epochs):
            train_loader = DataLoader(data, batch_size=self.ft_batch_size, shuffle=True)
            epoch_train_loss = []
            for x, y in train_loader:
                x, y = x.view(-1,50).to(self.device), y.to(self.device)
                _, y_hat = model(x)
                loss = self.loss_func(y, y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))

            print(f'{self.target_dataset} ft_train_epoch: {epoch} --- train_loss: {train_loss[-1]}')
        
        # Save the fine-tuned model and loss
        # 保存微调模型和损失
        model_path = os.path.join(self.ft_files, 'SSL_ft_model.pth')
        torch.save(model.state_dict(), model_path)
        loss_path = os.path.join(self.ft_files, 'SSL_ft_loss.npz')
        np.savez(loss_path, train_loss=train_loss)

    def test(self,X,true_c):
        """
        Test the model
        测试模型
        """
        model = Autoencoder().to(self.device)
        model_path = os.path.join(self.ft_files, f'SSL_ft_model.pth')
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        est_capacity = []
        for xi in X:
            xi = torch.tensor(xi, dtype=torch.float).to(self.device)
            xi = xi.view(1, 50)
            _, est_y = model(xi)
            est_y = est_y[0, 0]
            est_capacity.append(est_y.cpu().detach().numpy())

        # est_capacity = np.array(est_capacity)*C0 
        metrics = eval_metrix(true_c,est_capacity)
        return est_capacity,metrics