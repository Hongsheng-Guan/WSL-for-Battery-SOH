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


class PreTrain():
    def __init__(self,args) -> None:
        """
        Initialize the PreTrain class
        初始化 PreTrain 类
        """
        self.dataset = args.source_dataset
        self.epochs = args.pre_epochs
        self.batch_size = args.pre_batch_size
        self.lr = args.pre_lr
        self.save_folder = args.save_folder     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN_BiLSTM().to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.pre_lr)

    def train(self,X,Y):
        """
        Train the model
        训练模型
        """
        train_size = int(len(X)*0.7)
        val_size = len(X) - train_size
        data = MyDataset(X,Y)
        train_loss,val_loss = [],[]
        val_loss_crit = None
        # start_time = time.perf_counter()

        for epoch in range(self.epochs):
            # Split the dataset into training and validation sets
            # 将数据集分为训练集和验证集
            train_dataset, val_dataset = torch.utils.data.random_split(data,[train_size,val_size])
            train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)
            epoch_train_loss = []

            # Training loop
            # 训练循环
            for x,y in train_loader:
                x,y = x.to(self.device),y.to(self.device)
                y_hat = self.model(x)
                loss = self.loss_func(y,y_hat)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))

            # Validation loop
            # 验证循环
            with torch.no_grad():
                val_loader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=True)
                epoch_val_loss = []

                for x,y in val_loader:
                    x,y = x.to(self.device),y.to(self.device)
                    y_hat = self.model(x)
                    loss = self.loss_func(y,y_hat)
                    epoch_val_loss.append(loss.item())
                val_loss.append(np.mean(epoch_val_loss))

                # Save the model with the lowest validation loss
                # 保存验证损失最低的模型
                model_path = os.path.join(self.save_folder, 'pre_trained_model.pth')
                if val_loss_crit is None or val_loss_crit >= val_loss[-1]:
                    val_loss_crit = val_loss[-1]
                    torch.save(self.model.state_dict(), model_path)

            print(f'{self.dataset} pre_train_epoch: {epoch} --- train_loss: {train_loss[-1]} --- val_loss: {val_loss[-1]} --- val_loss_crit: {val_loss_crit}')
        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # print(f"Total pre-training time: {total_time:.4f} seconds")

        loss_path = os.path.join(self.save_folder, 'pre_trained_loss.npz')
        np.savez(loss_path, train_loss=train_loss, val_loss=val_loss)


class FtTrain():
    def __init__(self,args) -> None:
        """
        Initialize the FtTrain class
        初始化 FtTrain 类
        """
        self.source_dataset = args.source_dataset
        self.target_dataset = args.target_dataset
        self.epochs = args.ft_epochs
        self.batch_size = args.ft_batch_size
        self.lr = args.ft_lr
        self.ft_files = args.ft_files
        self.pre_model_file = args.pre_model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN_BiLSTM().to(self.device)
        self.loss_func = nn.MSELoss()

    def train(self,X,Y):
        """
        Train the model
        训练模型
        """
        data = MyDataset(X,Y)
        pre_model_path = os.path.join(self.pre_model_file, 'pre_trained_model.pth')
        self.model.load_state_dict(torch.load(pre_model_path, weights_only=True))
        # Freeze the pre-trained layers
        # 冻结预训练层
        for name, parameter in self.model.named_parameters():
            if 'lstm' in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

        # freeze Conv only
        # for name, parameter in self.model.named_parameters():
        #     if 'cnn' in name:
        #         parameter.requires_grad = False
        #     else:
        #         parameter.requires_grad = True

        # freeze FC only
        # for name, parameter in self.model.named_parameters():
        #     if 'fc' in name:
        #         parameter.requires_grad = False
        #     else:
        #         parameter.requires_grad = True

        # unfreeze last FC layer
        # for name, parameter in self.model.named_parameters():
        #     if 'fc2' in name:
        #         parameter.requires_grad = True
        #     else:
        #         parameter.requires_grad = False

        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        train_loss = []

        # start_time = time.perf_counter()

        for epoch in range(self.epochs):
            train_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
            epoch_train_loss = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.loss_func(y, y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))

            print(f'{self.target_dataset} ft_train_epoch: {epoch} --- train_loss: {train_loss[-1]}')
        
        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # print(f"Total fine-tuning time: {total_time:.4f} seconds")
        
        # Save the fine-tuned model and loss
        # 保存微调模型和损失
        model_path = os.path.join(self.ft_files, f'from_{self.source_dataset}_ft_model.pth')
        torch.save(self.model.state_dict(), model_path)
        loss_path = os.path.join(self.ft_files, f'from_{self.source_dataset}_ft_loss.npz')
        np.savez(loss_path, train_loss=train_loss)


class Test():
    def __init__(self,args) -> None:
        """
        Initialize the Test class
        初始化 Test 类
        """
        self.source_dataset = args.source_dataset
        self.ft_files= args.ft_files
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test(self,X,true_c):
        """
        Test the model
        测试模型
        """
        model = CNN_BiLSTM().to(self.device)
        model_path = os.path.join(self.ft_files, f'from_{self.source_dataset}_ft_model.pth')
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # start_time = time.perf_counter()
        est_capacity = []
        for xi in X:
            xi = torch.tensor(xi, dtype=torch.float).to(self.device)
            xi = xi.view(1, 50, 1)
            est_y = model(xi)
            est_y = est_y[0, 0]
            est_capacity.append(est_y.cpu().detach().numpy())
        
        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # print(f"Total testing time: {total_time:.4f} seconds")
        metrics = eval_metrix(true_c,est_capacity)
        return est_capacity,metrics