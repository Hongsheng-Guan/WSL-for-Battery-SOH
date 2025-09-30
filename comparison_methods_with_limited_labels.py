from data_process.data_load import load_pre_data, load_ft_data, load_test_data
from train.CNN_train import CNNTrain
from train.SSL_train import SSLTrain
from train.Benchmark_train import BenchmarkTrain
import argparse
import pandas as pd
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
    
    # Target datasets
    # 目标数据集
    parser.add_argument('--target_dataset', type=list, default=['Dataset 1','Dataset 2','Dataset 3','Dataset 4','Dataset 5','Dataset 6'], help='The name of target datasets')
    parser.add_argument('--ft_data_num',type=int,default=6,help='Numbers of fine-tuning samples')
    args = parser.parse_args()

    return args

def cnn_one_experiment(args):
    """
    Conduct one experiment(only for CNN model)
    进行一次实验(仅适用于CNN模型)
    """
    train_data = load_ft_data(args)
    trainer = CNNTrain(args)
    trainer.train(train_data['X'],train_data['Y'])

    df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
    test_cells = df['test_cells'].values
    
    eval_metrics = {}
    test_results = {}
    
    for test_cell in test_cells:
        if str(test_cell)=='nan':
            continue
        setattr(args, 'test_cell', test_cell)
        
        test_data = load_test_data(args.target_dataset, test_cell)
        true_capacity = test_data['Y']
        est_capacity, metric = trainer.test(test_data['X'], true_capacity)
        
        test_results[f'{test_cell}_est'] = est_capacity
        test_results[f'{test_cell}_true'] = true_capacity
        eval_metrics[test_cell] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def cnn_results():
    """
    Main function to run experiments(only for CNN model)
    运行每个数据集的主函数(仅适用于CNN模型)
    """
    args = get_args()
    target_datasets = args.target_dataset
    for target_dataset in target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        model_file = os.path.join('results', 'comparison_methods_with_limited_labels_results','CNN', target_dataset)
        setattr(args, 'save_folder', model_file)

        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        train_cells = df['ft_cells'].values

        for i,cell in enumerate(train_cells):
            if pd.isna(cell):
                continue

            args.ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_cell', cell)
            if not os.path.exists(args.ft_files ):
                os.makedirs(args.ft_files )
            
            cnn_one_experiment(args)

def SSL_one_experiment(args):
    """
    Conduct one experiment(only for AE model)
    进行一次实验(仅适用于AE模型)
    """
    print(f"Fine-tuning the AE ({args.source_dataset}) for {args.ft_cell} ({args.target_dataset})")
    ft_data = load_ft_data(args)
    trainer = SSLTrain(args)
    trainer.ft_train(ft_data['X'],ft_data['Y'])

    df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
    test_cells = df['test_cells'].values
    
    eval_metrics = {}
    test_results = {}
    
    for test_cell in test_cells:
        if str(test_cell)=='nan':
            continue
        setattr(args, 'test_cell', test_cell)
        
        test_data = load_test_data(args.target_dataset, test_cell)
        true_capacity = test_data['Y']
        est_capacity, metric = trainer.test(test_data['X'], true_capacity)
        
        test_results[f'{test_cell}_est'] = est_capacity
        test_results[f'{test_cell}_true'] = true_capacity
        eval_metrics[test_cell] = metric
    
    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def SSL_results():
    """
    Main function to run experiments(only for AE model)
    运行每个数据集的主函数(仅适用于AE模型)
    """
    args = get_args()
    setattr(args,'pre_data_rate','all')
    target_datasets = args.target_dataset

    for target_dataset in target_datasets:
        # Set save folder and dataset attributes
        # 设置保存文件夹和数据集属性
        save_folder = os.path.join('results', 'comparison_methods_with_limited_labels_results','SSL', target_dataset)
        setattr(args, 'save_folder', save_folder)
        setattr(args, 'source_dataset', target_dataset) # The source dataset is the same as the target dataset
        setattr(args, 'target_dataset', target_dataset) 
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        # Pre-train the model
        # 预训练模型    
        print('Pre-training the model')
        pre_data = load_pre_data(args)
        setattr(args, 'ft_files', args.save_folder)
        trainer = SSLTrain(args)
        trainer.pre_train(pre_data['X'], pre_data['Y'])

        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        ft_cells = df['ft_cells'].values

        # Conduct one experiment for each fine-tuning cell
        for i,cell in enumerate(ft_cells):
            if str(cell)=='nan':
                continue

            ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_files', ft_files)
            setattr(args, 'ft_cell', cell)
            if not os.path.exists(args.ft_files):
                os.makedirs(args.ft_files)

            SSL_one_experiment(args)
    
def Benchmark_one_experiment(args):
    """
    Conduct one experiment(only for Benchmark model)
    进行一次实验(仅适用于无预训练Benchmark模型)
    """
    train_data = load_ft_data(args)
    trainer = BenchmarkTrain(args)
    trainer.train(train_data['X'],train_data['Y'])

    df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
    test_cells = df['test_cells'].values
    
    eval_metrics = {}
    test_results = {}
    
    for test_cell in test_cells:
        if str(test_cell)=='nan':
            continue
        setattr(args, 'test_cell', test_cell)
        test_data = load_test_data(args.target_dataset, test_cell)
        
        true_capacity = test_data['Y']
        est_capacity, metric = trainer.test(test_data['X'], true_capacity)
        
        test_results[f'{test_cell}_est'] = est_capacity
        test_results[f'{test_cell}_true'] = true_capacity
        eval_metrics[test_cell] = metric
    
    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv', index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)


def Benchmark_results():
    """
    Main function to run experiments(only for Benchmark model)
    运行每个数据集的主函数(仅适用于无预训练Benchmark模型)
    """
    args = get_args()
    target_datasets = args.target_dataset
    for target_dataset in target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        model_file = os.path.join('results', 'comparison_methods_with_limited_labels_results','Benchmark', target_dataset)
        setattr(args, 'save_folder', model_file)

        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        train_cells = df['ft_cells'].values

        for i,cell in enumerate(train_cells):
            if pd.isna(cell):
                continue

            args.ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_cell', cell)
            if not os.path.exists(args.ft_files ):
                os.makedirs(args.ft_files )
            
            Benchmark_one_experiment(args)

def rf_results():
    args = get_args()
    target_datasets = args.target_dataset
    for target_dataset in target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        model_file = os.path.join('results', 'comparison_methods_with_limited_labels_results','RF', target_dataset)
        setattr(args, 'save_folder', model_file)

        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        train_cells = df['ft_cells'].values

        for i,cell in enumerate(train_cells):
            if pd.isna(cell):
                continue

            args.ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_cell', cell)
            if not os.path.exists(args.ft_files):
                os.makedirs(args.ft_files)
            
            train_data = load_ft_data(args)
            train_x = train_data['X']
            train_y = train_data['Y']
            rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=9)
            rf.fit(train_x, train_y)

            df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
            test_cells = df['test_cells'].values
            
            eval_metrics = {}
            test_results = {}
            
            for test_cell in test_cells:
                if str(test_cell)=='nan':
                    continue
                setattr(args, 'test_cell', test_cell)
                
                test_data = load_test_data(args.target_dataset, test_cell)
                test_x = test_data['X']
                true_capacity = test_data['Y']
                est_capacity = rf.predict(test_x)
                metric = eval_metrix(true_capacity, est_capacity)
                
                test_results[f'{test_cell}_est'] = est_capacity
                test_results[f'{test_cell}_true'] = true_capacity
                eval_metrics[test_cell] = metric
            
            test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
            test_results_df.to_csv(f'{args.ft_files}/test_results.csv', index=False)
            
            eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
            eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def gpr_results():
    args = get_args()
    target_datasets = args.target_dataset
    for target_dataset in target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        model_file = os.path.join('results', 'comparison_methods_with_limited_labels_results','GPR', target_dataset)
        setattr(args, 'save_folder', model_file)

        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        train_cells = df['ft_cells'].values

        for i,cell in enumerate(train_cells):
            if pd.isna(cell):
                continue

            args.ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_cell', cell)
            if not os.path.exists(args.ft_files):
                os.makedirs(args.ft_files)
            
            train_data = load_ft_data(args)
            train_x = train_data['X']
            train_y = train_data['Y']
            kernel = Matern(nu=2.5)
            gpr = GaussianProcessRegressor(kernel=kernel,alpha=1e-5,n_restarts_optimizer=3,random_state=9)
            gpr.fit(train_x, train_y)

            df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
            test_cells = df['test_cells'].values
            
            eval_metrics = {}
            test_results = {}
            
            for test_cell in test_cells:
                if str(test_cell)=='nan':
                    continue
                setattr(args, 'test_cell', test_cell)
                
                test_data = load_test_data(args.target_dataset, test_cell)
                test_x = test_data['X']
                true_capacity = test_data['Y']
                est_capacity = gpr.predict(test_x)
                metric = eval_metrix(true_capacity, est_capacity)
                
                test_results[f'{test_cell}_est'] = est_capacity
                test_results[f'{test_cell}_true'] = true_capacity
                eval_metrics[test_cell] = metric
            
            test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
            test_results_df.to_csv(f'{args.ft_files}/test_results.csv', index=False)
            
            eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
            eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def svr_results():
    args = get_args()
    target_datasets = args.target_dataset
    for target_dataset in target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        model_file = os.path.join('results', 'comparison_methods_with_limited_labels_results','SVR', target_dataset)
        setattr(args, 'save_folder', model_file)

        df = pd.read_csv(os.path.join('data', target_dataset, 'train_cells_id.csv'))
        train_cells = df['ft_cells'].values

        for i,cell in enumerate(train_cells):
            if pd.isna(cell):
                continue

            args.ft_files = os.path.join(args.save_folder, f'Experiment{i+1}({cell})')
            setattr(args, 'ft_cell', cell)
            if not os.path.exists(args.ft_files):
                os.makedirs(args.ft_files)
            
            train_data = load_ft_data(args)
            train_x = train_data['X']
            train_y = train_data['Y']
            svr = SVR(kernel='rbf',epsilon=0.001)
            svr.fit(train_x, train_y)

            df = pd.read_csv(f'data/{args.target_dataset}/train_cells_id.csv')
            test_cells = df['test_cells'].values
            
            eval_metrics = {}
            test_results = {}
            
            for test_cell in test_cells:
                if str(test_cell)=='nan':
                    continue
                setattr(args, 'test_cell', test_cell)
                
                test_data = load_test_data(args.target_dataset, test_cell)
                test_x = test_data['X']
                true_capacity = test_data['Y']
                est_capacity = svr.predict(test_x)
                metric = eval_metrix(true_capacity, est_capacity)
                
                test_results[f'{test_cell}_est'] = est_capacity
                test_results[f'{test_cell}_true'] = true_capacity
                eval_metrics[test_cell] = metric
            
            test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
            test_results_df.to_csv(f'{args.ft_files}/test_results.csv', index=False)
            
            eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
            eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

if __name__ == '__main__':
    # cnn_results()
    # SSL_results()
    # Benchmark_results()
    # rf_results()
    # gpr_results()
    # svr_results()
    pass
    