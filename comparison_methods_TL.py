from data_process.data_load import normalization,load_ft_data,load_test_data
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
    parser = argparse.ArgumentParser('Hyper Parameters')

    # Datasets
    # 数据集
    parser.add_argument('--source_dataset', type=str, default='Dataset 6', help='the name of source_dataset')
    parser.add_argument('--target_dataset', type=list, default=['Dataset 1','Dataset 2','Dataset 3','Dataset 4','Dataset 5'], help='the name of target_datasets')
    
    # 选择测试模型
    # Choose test model
    # parser.add_argument('--test_model', type=str, default='TL', help='Model name')
    parser.add_argument('--test_model', type=str, default='WSL', help='Model name')

    # Pre-training parameters
    # 预训练参数
    parser.add_argument('--pre_epochs', type=int, default=300, help='Epochs for pre-training')
    parser.add_argument('--pre_batch_size', type=int, default=1024, help='Batch size for pre-training')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='Learning rate for pre-training')

    # Fine-tuning parameters
    # 微调参数
    parser.add_argument('--ft_epochs', type=int, default=50, help='Epochs for fine-tuning')
    parser.add_argument('--ft_batch_size', type=int, default=4, help='Batch size for fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=0.0005, help='Learning rate for fine-tuning')
    parser.add_argument('--ft_data_num',type=int,default=6,help='Numbers of fine-tuning sameples')

    args = parser.parse_args()
    return args

def get_pre_data(args):
    """
    Load pre-training data based on the source dataset
    根据源数据集加载预训练数据
    """
    data_path = os.path.join('data',args.source_dataset)
    unlabeled_cells = None
    if args.source_dataset == 'Dataset 6':
        unlabeled_cells = ['25_0.5a_80','25_0.5b_80','25_0.5c_80','25_0.5d_80','25_0.5a_60','25_0.5b_60','25_3a_80','25_3b_80','25_3a_60','25_3b_60']
    train_x,train_y = [], []
    data_files = os.listdir(data_path)
    for fi in data_files:
        if fi[:-4] in unlabeled_cells:
            continue
        if 'npz' not in fi:
            continue
        file_path = os.path.join(data_path,fi)
        cell_data = np.load(file_path) # Q_sequences,weak_label,capacity
        cell_x,cell_y = None,None
        if args.test_model == 'TL':
            cell_x,cell_y = normalization(cell_data['Q_sequences'],cell_data['capacity'])
        elif args.test_model == 'WSL':
            cell_x,cell_y = normalization(cell_data['Q_sequences'],cell_data['weak_label'])
        train_x.append(cell_x)
        train_y.append(cell_y)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    return train_x, train_y

def one_experiment(args):
    """
    Conduct one experiment
    进行一次实验
    """
    # Load fine-tuning data
    # 加载微调数据
    ft_data = load_ft_data(args)

    # Initialize and train fine-tuning model
    # 初始化并训练微调模型
    print(f"Fine-tuning the model ({args.source_dataset}) for {args.ft_cell} ({args.target_dataset})")
    ft_train = FtTrain(args)
    ft_train.train(ft_data['X'],ft_data['Y'])

    # Read training cell IDs from target dataset
    # 读取目标数据集的训练单元ID
    df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
    test_cells = df['test_cells'].values
    
    eval_metrics = {}
    test_results = {}
    
    for test_cell in test_cells:
        if str(test_cell)=='nan':
            continue
        setattr(args, 'test_cell', test_cell)
        
        # Load test data
        # 加载测试数据
        test_data = load_test_data(args.target_dataset, test_cell)
        
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
    Main function to run experiments for each target dataset
    主函数，针对每个目标数据集运行实验
    """
    args = get_args()
    target_datasets = args.target_dataset
    
    save_folder = os.path.join('results', 'comparison_methods_TL_results',args.test_model)
    setattr(args, 'save_folder', save_folder)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Load pre-training data (TL: all labeled data from source dataset. WSL: all unlabeled data from source dataset)
    # 加载预训练数据
    train_x,train_y = get_pre_data(args)
    
    # Pre-train the model
    # 预训练模型
    pre_train = PreTrain(args)
    pre_train.train(train_x,train_y)
    setattr(args, 'pre_model_file', save_folder)

    for target_dataset in target_datasets:
        setattr(args,'target_dataset',target_dataset)
        setattr(args, 'save_folder', os.path.join(save_folder,f'{target_dataset} transfer from {args.source_dataset}'))
        # Load fine-tuning cell IDs
        # 加载微调单元ID 
        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        ft_cells = df['ft_cells'].values

        for i,cell in enumerate(ft_cells):
            if pd.isna(cell):
                continue

            ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args,'ft_files',ft_files)
            setattr(args,'ft_cell',cell)

            # Create fine-tuning files folder if it does not exist
            # 如果微调文件夹不存在，则创建
            if not os.path.exists(args.ft_files):
                os.makedirs(args.ft_files)
            
            # Conduct one experiment
            # 进行一次实验
            one_experiment(args)

if __name__ == '__main__':
    # main()
    pass
