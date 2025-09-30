from data_process.data_load import load_pre_data,load_ft_data,load_test_data
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
    parser.add_argument('--target_dataset', type=list, default=['Dataset 1','Dataset 2','Dataset 3', 'Dataset 4', 'Dataset 5', 'Dataset 6'], help='The name of target datasets')
    
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

def one_experiment(args):
    """
    Conduct one experiment
    进行一次实验
    """
    # Load fine-tuning data
    # 加载微调数据
    ft_data = load_ft_data(args)

    # Fine-tune the model
    # 模型微调
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
    Main function to run experiments
    运行每个数据集的主函数
    """
    args = get_args()
    target_datasets = args.target_dataset

    for target_dataset in target_datasets:
        # Set save folder and dataset attributes
        # 设置保存文件夹和数据集属性
        save_folder = os.path.join('results', 'soh_individual_dataset_results', target_dataset)
        setattr(args, 'save_folder', save_folder)
        setattr(args, 'source_dataset', target_dataset)
        setattr(args, 'target_dataset', target_dataset) 

        # Create save folder if it does not exist
        # 如果保存文件夹不存在，则创建
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        # Load pre-training data
        # 加载预训练数据
        pre_data = load_pre_data(args)
        
        # Pre-train the model
        # 预训练模型
        print('Pre-training the model')
        pre_train = PreTrain(args)
        pre_train.train(pre_data['X'], pre_data['Y'])
        
        # Set the pre-trained model file attribute
        # 设置预训练模型文件属性
        setattr(args, 'pre_model_file', save_folder)

        # Load fine-tuning cell IDs
        # 加载微调单元ID
        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        ft_cells = df['ft_cells'].values

        # Conduct one experiment for each fine-tuning cell
        for i,cell in enumerate(ft_cells):
            if str(cell)=='nan':
                continue

            # Set fine-tuning files and cell attributes
            # 设置微调文件和单元属性
            ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_files', ft_files)
            setattr(args, 'ft_cell', cell)
            
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
    