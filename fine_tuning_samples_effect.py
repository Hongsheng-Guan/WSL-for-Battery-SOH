from data_process.data_load import load_pre_data,load_ft_data,load_test_data
from train.WSL_train import PreTrain,FtTrain,Test
import argparse
import pandas as pd
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters')
    parser.add_argument('--datasets', type=list, default=['Dataset 1','Dataset 2','Dataset 3','Dataset 4','Dataset 5','Dataset 6'], help='The name of dataset')
    
    # fine-tuning related
    parser.add_argument('--ft_epochs', type=int, default=50, help='Epochs for fine-tuning')
    parser.add_argument('--ft_batch_size', type=int, default=4, help='Batch size for fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=0.0005, help='Learning rate for fine-tuning')
    parser.add_argument('--ft_data_num',type=list,default=[2+i for i in range(0,10)]+['all'],help='Numbers of fine-tuning samples')

    # test related
    parser.add_argument('--test_model', type=str, default='CNN_BiLSTM', help='The name of our model')
    
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
    args = get_args()
    all_datasets = args.datasets
    ft_data_num = args.ft_data_num
    if ft_data_num == 'all':
        args.ft_batch_size = 64
    for dataset in all_datasets:
        setattr(args,'pre_model_file', os.path.join('results','soh_individual_dataset_results',dataset))
        setattr(args,'source_dataset',dataset)
        setattr(args,'target_dataset',dataset)
        for data_num in ft_data_num:
            save_folder = os.path.join('results','fine_tuning_samples_effect_results',dataset,f'ft_num={str(data_num)}')
            setattr(args,'ft_data_num',data_num)
            setattr(args,'save_folder',save_folder)
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

            df = pd.read_csv(os.path.join('data', args.target_dataset, 'train_cells_id.csv'))
            ft_cells = df['ft_cells'].values
            # print(dataset,args.ft_data_num)
            for i,cell in enumerate(ft_cells):
                if str(cell)=='nan':
                    continue
                ft_files = os.path.join(args.save_folder,f'Experiment{i+1}({cell})')
                setattr(args,'ft_files',ft_files)
                setattr(args,'ft_cell',cell)
                if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
                one_experiment(args)

if __name__ == '__main__':
    # main()
    pass
    