from data_process.data_load import load_ft_data,load_test_data
from train.WSL_train import FtTrain,Test
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

    # Target datasets
    # 目标数据集
    parser.add_argument('--target_dataset', type=list, default=['Dataset 1','Dataset 2','Dataset 3','Dataset 4','Dataset 5','Dataset 6'], help='the name of target_datasets')

    # Fine-tuning parameters
    # 微调参数
    parser.add_argument('--ft_epochs', type=int, default=50, help='Epochs for fine-tuning')
    parser.add_argument('--ft_batch_size', type=int, default=4, help='Batch size for fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=0.0005, help='Learning rate for fine-tuning')
    parser.add_argument('--ft_data_num',type=int,default=6,help='Numbers of fine-tuning sameples')

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
    Main function to run experiments for each target dataset
    主函数，针对每个目标数据集运行实验
    """
    args = get_args()
    target_datasets = args.target_dataset
    source_datasets = args.target_dataset

    for target_dataset in target_datasets:
        setattr(args,'target_dataset',target_dataset)
        if target_dataset !='Dataset 3':
            continue

        for source_dataset in source_datasets:
            # Skip if the target dataset is the same as the source dataset
            # 如果目标数据集与源数据集相同，则跳过
            if target_dataset == source_dataset:
                continue
            # Pre-trained model is the same as in the soh_individual_dataset experiment
            # 预训练模型与cross_condition实验中相同
            pre_model_file = os.path.join('results', 'soh_individual_dataset_results', source_dataset)
            setattr(args, 'pre_model_file', pre_model_file)

            setattr(args,'source_dataset',source_dataset)
            save_folder = os.path.join('results', 'soh_cross_laboratory_datasets_results', target_dataset, f'transfer from {source_dataset}')
            setattr(args, 'save_folder', save_folder)

            # Create save folder if it does not exist
            # 如果保存文件夹不存在，则创建
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

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
