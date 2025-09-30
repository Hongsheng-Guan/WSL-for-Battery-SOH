from train.WSL_train import PreTrain,FtTrain,Test
import argparse
import pandas as pd
import numpy as np
import os

def get_args():
    """
    Parse command line arguments
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Hyper Parameters')
    
    # Target datasets
    # 目标数据集
    parser.add_argument('--target_dataset', type=str, default='Dataset 3', help='The name of target datasets')
    
    # Pre-training parameters
    # 预训练参数
    parser.add_argument('--pre_epochs', type=int, default=300, help='Epochs for pre-training')
    parser.add_argument('--pre_batch_size', type=int, default=1024, help='Batch size for pre-training')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='Learning rate for pre-training')
    parser.add_argument('--pre_data_rate', type=str, default='all', help='Rate of pre-training data')

    # Fine-tuning parameters
    # 微调参数
    parser.add_argument('--ft_epochs', type=int, default=50, help='Epochs for fine-tuning')
    parser.add_argument('--ft_batch_size', type=int, default=4, help='Batch size for fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=0.0005, help='Learning rate for fine-tuning')
    parser.add_argument('--ft_data_num', type=int, default=6, help='Numbers of fine-tuning samples')

    args = parser.parse_args()

    return args

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

def load_pre_data(path):
    """
    Load pre-training data
    加载预训练数据
    """
    df = pd.read_csv(f'data/Dataset 3/train_cells_id.csv')
    pre_cells = df['pre_cells'].values
    X,Y = [],[]

    for cell in pre_cells:
        if pd.isna(cell):
            continue
        file_path = path+f'/{cell}.npz'
        if not os.path.exists(file_path):
            continue
        cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
        cell_x,cell_y = normalization(cell_data['Q_sequences'],cell_data['weak_label'])
        X.append(cell_x)
        Y.append(cell_y)

    X,Y = np.concatenate(X,axis=0),np.concatenate(Y,axis=0)
    data = {
        'X':X,
        'Y':Y
    }

    return data

def load_ft_data(args,path):
    """
    Load fine-tuning data
    加载微调数据
    """
    file_path = path+f'/{args.ft_cell}.npz'
    cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
    X,Y = normalization(cell_data['Q_sequences'],cell_data['capacity'])

    data = {}
    if args.ft_data_num == 'all':
        data['X'],data['Y'] = X,Y
    else:
        sparse_ids = np.linspace(0, len(X) - 1, num=args.ft_data_num, endpoint=True, dtype=int)
        data['X'],data['Y'] =X[sparse_ids],Y[sparse_ids]

    return data

def load_test_data(path,cell):
    """
    Load fine-tuning data
    加载测试数据
    """
    file_path = path+f"/{cell}.npz"
    cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
    data = {}
    data['X'],data['Y'] = normalization(cell_data['Q_sequences'],cell_data['capacity'])
    return data


def one_experiment(args,path):
    """
    Conduct one experiment
    进行一次实验
    """
    # Load fine-tuning data
    # 加载微调数据
    ft_data = load_ft_data(args,path)

    # Fine-tune the model
    # 模型微调
    print(f"Fine-tuning the model ({args.source_dataset}) for {args.ft_cell} ({args.target_dataset})")
    ft_train = FtTrain(args)
    ft_train.train(ft_data['X'],ft_data['Y'])

    # Read training cell IDs from target dataset
    # 读取目标数据集的训练单元ID
    df = pd.read_csv(f'data/Dataset 3/train_cells_id.csv')
    test_cells = df['test_cells'].values
    
    eval_metrics = {}
    test_results = {}
    
    for test_cell in test_cells:
        if str(test_cell)=='nan':
            continue
        setattr(args, 'test_cell', test_cell)
        
        # Load test data
        # 加载测试数据
        test_data = load_test_data(path, test_cell)
        
        # Initialize and test model
        # 初始化并测试模型
        tester = Test(args)
        true_capacity = test_data['Y']
        est_capacity, metric = tester.test(test_data['X'], true_capacity)
        
        test_results[f'{test_cell}_est'] = est_capacity
        test_results[f'{test_cell}_true'] = true_capacity
        eval_metrics[test_cell] = metric
    
    # Save test results and evaluation metrics
    # 保存测试结果和评估指标
    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def main():
    """
    Main function to run experiments
    运行每个数据集的主函数
    """
    args = get_args()
    setattr(args, 'source_dataset', args.target_dataset)
    setattr(args, 'target_dataset', args.target_dataset)

    start_vols = np.linspace(3.3,3.95,14)
    start_vols = np.round(start_vols,2)
    for i in range(len(start_vols)):
        l_vol = start_vols[i]
        r_vol = np.linspace(l_vol+0.1,4.05,14-i)
        for j in range(len(r_vol)):
            r_vol[j] = round(r_vol[j],2)
            setattr(args, 'save_folder', os.path.join('results', 'voltage_window_effect',f'V({l_vol}-{r_vol[j]})'))
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

            path = f'data/Dataset 3 with different V window/V({l_vol}-{r_vol[j]})'
            # pre_data = load_pre_data(path)
            # pre_train = PreTrain(args)
            # pre_train.train(pre_data['X'], pre_data['Y'])
            setattr(args, 'pre_model_file', args.save_folder)

            df = pd.read_csv(os.path.join('data', args.target_dataset, 'train_cells_id.csv'))
            ft_cells = df['ft_cells'].values

            for i,cell in enumerate(ft_cells):
                if str(cell)=='nan':
                    continue
                ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
                setattr(args, 'ft_files', ft_files)
                setattr(args, 'ft_cell', cell)
                if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
                one_experiment(args,path)

if __name__ == '__main__':
    main()
    # pass
    