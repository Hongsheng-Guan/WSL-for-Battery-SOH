from data_process.data_load import load_ft_data, load_test_data
from train.CNN_train import CNNTrain
from train.WSL_train import FtTrain,Test
from train.Benchmark_train import BenchmarkTrain
import argparse
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tool.metrix import eval_metrix
import os


def get_args():
    """
    Parse command line arguments
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Hyper Parameters')

    parser.add_argument('--target_datasets', type=list, default=['Dataset 1','Dataset 2','Dataset 3', 'Dataset 4', 'Dataset 5', 'Dataset 6'], help='The name of target dataset')
    parser.add_argument('--ft_data_num',type=str,default='all',help='Numbers of training samples')
    args = parser.parse_args()
    return args

def get_train_data(args):
    train_cells = {
        'Dataset 1': ['25-1','25-2'],
        'Dataset 2': ['1C-4','1C-5'],
        'Dataset 3': ['0-CC-1'],
        'Dataset 4': ['CY25-05_1-#14','CY25-05_1-#16'],
        'Dataset 5': ['2C-6','2C-7'],
        'Dataset 6': ['25_1a_100']
    }

    train_x,train_y = [],[]
    di_train_cells = train_cells[args.target_dataset]
    for train_ci in di_train_cells:
        if str(train_ci)=='nan':
            continue
        setattr(args, 'ft_cell', train_ci)
        data_ci = load_ft_data(args)
        train_x.append(data_ci['X'])
        train_y.append(data_ci['Y'])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    return train_x, train_y

def cnn_results():
    args = get_args()
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.target_dataset,'CNN')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
            os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        trainer = CNNTrain(args,batch_size=128)
        trainer.train(train_x,train_y)

        df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
        test_cells = df['test_cells'].values
        
        eval_metrics = {}
        test_results = {}
        for test_ci in test_cells:
            if str(test_ci)=='nan':
                continue
            test_data = load_test_data(args.target_dataset,test_ci)
            true_capacity = test_data['Y']
            est_capacity, metric = trainer.test(test_data['X'], true_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
    
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
        eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def Benchmark_results():
    args = get_args()
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results', args.target_dataset,'Benchmark')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
            os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        trainer = BenchmarkTrain(args,batch_size=128)
        trainer.train(train_x,train_y)

        df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
        test_cells = df['test_cells'].values
        
        eval_metrics = {}
        test_results = {}
        for test_ci in test_cells:
            if str(test_ci)=='nan':
                continue
            test_data = load_test_data(args.target_dataset,test_ci)
            true_capacity = test_data['Y']
            est_capacity, metric = trainer.test(test_data['X'], true_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
    
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
        eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def WSL_el_results():
    args = get_args()
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.target_dataset,'WSL_el')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
            os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)

        setattr(args,'source_dataset',args.target_dataset)
        setattr(args,'ft_epochs',50)
        setattr(args,'ft_batch_size',128)
        setattr(args,'ft_lr',0.0005)
        setattr(args, 'pre_model_file', f'results/soh_individual_dataset_results/{args.target_dataset}')
        ft_train = FtTrain(args)
        ft_train.train(train_x,train_y)

        df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
        test_cells = df['test_cells'].values
        
        eval_metrics = {}
        test_results = {}
        tester = Test(args)
        for test_ci in test_cells:
            if str(test_ci)=='nan':
                continue
            test_data = load_test_data(args.target_dataset,test_ci)
            true_capacity = test_data['Y']
            est_capacity, metric = tester.test(test_data['X'], true_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
    
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
        eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def rf_results():
    args = get_args()
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.target_dataset,'RF')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
            os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=9)
        rf.fit(train_x, train_y)

        df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
        test_cells = df['test_cells'].values
        
        eval_metrics = {}
        test_results = {}
        for test_ci in test_cells:
            if str(test_ci)=='nan':
                continue
            test_data = load_test_data(args.target_dataset,test_ci)
            test_x = test_data['X']
            true_capacity = test_data['Y']
            est_capacity = rf.predict(test_x)
            metric = eval_metrix(true_capacity, est_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
    
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
        eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def svr_results():
    args = get_args()
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results', args.target_dataset,'SVR')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
            os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        svr = SVR(kernel='rbf',epsilon=0.001)
        svr.fit(train_x, train_y)

        df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
        test_cells = df['test_cells'].values
        
        eval_metrics = {}
        test_results = {}
        for test_ci in test_cells:
            if str(test_ci)=='nan':
                continue
            test_data = load_test_data(args.target_dataset,test_ci)
            test_x = test_data['X']
            true_capacity = test_data['Y']
            est_capacity = svr.predict(test_x)
            metric = eval_metrix(true_capacity, est_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
    
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
        eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

if __name__ == '__main__':
    # cnn_results()
    # Benchmark_results()
    # rf_results()
    # svr_results()
    # WSL_el_results()
    pass
    