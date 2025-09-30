import numpy as np
import pandas as pd
import os

def normalization(x,y=None):
    """
    Normalize the input data
    归一化输入数据
    """
    x = x/x[0][-1]
    if y is None:
        return x
    else:
        y = y/y[0]
        return x,y

def load_pre_data(args):
    """
    Load pre-training data
    加载预训练数据
    """
    df = pd.read_csv(f'data/{args.source_dataset}/train_cells_id.csv')
    pre_cells = df['pre_cells'].values
    X,Y = [],[]

    for cell in pre_cells:
        if pd.isna(cell):
            continue

        file_path = f'data/{args.source_dataset}/{cell}.npz'
        if not os.path.exists(file_path):
            continue

        cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
        cell_x,cell_y = normalization(cell_data['Q_sequences'],cell_data['weak_label'])
        X.append(cell_x)
        Y.append(cell_y)

    X,Y = np.concatenate(X,axis=0),np.concatenate(Y,axis=0)
    data = {}

    if args.pre_data_rate == 'all':
        data['X'] = X
        data['Y'] = Y
    else:
        num = len(X)
        random_ids = np.random.randint(0,num,int(num*args.pre_data_rate))
        data['X'],data['Y'] = X[random_ids],Y[random_ids]

    return data

def load_ft_data(args):
    """
    Load fine-tuning data
    加载微调数据
    """
    file_path = f'data/{args.target_dataset}/{args.ft_cell}.npz'
    cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
    X,Y = normalization(cell_data['Q_sequences'],cell_data['capacity'])

    data = {}
    if args.ft_data_num == 'all':
        data['X'],data['Y'] = X,Y
    else:
        sparse_ids = np.linspace(0, len(X) - 1, num=args.ft_data_num, endpoint=True, dtype=int)
        data['X'],data['Y'] =X[sparse_ids],Y[sparse_ids]

    return data

def load_test_data(dataset,cell):
    """
    Load fine-tuning data
    加载测试数据
    """
    file_path = f"data/{dataset}/{cell}.npz"
    cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
    data = {}
    data['X'],data['Y'] = normalization(cell_data['Q_sequences'],cell_data['capacity'])
    return data
