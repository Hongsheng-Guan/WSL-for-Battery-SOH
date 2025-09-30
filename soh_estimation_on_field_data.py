from data_process.data_load import load_pre_data,load_ft_data,load_test_data
from train.WSL_train import PreTrain,FtTrain,Test
from train.CNN_train import CNNTrain
from train.SSL_train import SSLTrain
from train.Benchmark_train import BenchmarkTrain
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tool.metrix import eval_metrix
import argparse
import pandas as pd
import numpy as np
import random
import os

def get_args():
    """
    Parse command line arguments
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Hyper Parameters')
    
    # Target datasets
    # 目标数据集
    parser.add_argument('--target_dataset', type=str, default='Dataset 8', help='The name of target datasets')

    # Pre-training parameters
    # 预训练参数
    parser.add_argument('--pre_epochs', type=int, default=300, help='Epochs for pre-training')
    parser.add_argument('--pre_batch_size', type=int, default=1024, help='Batch size for pre-training')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='Learning rate for pre-training')
    parser.add_argument('--pre_data_rate', type=str, default='all', help='Rate of pre-training data')

    # Fine-tuning parameters
    # 微调参数
    parser.add_argument('--ft_epochs', type=int, default=200, help='Epochs for fine-tuning')
    parser.add_argument('--ft_batch_size', type=int, default=4, help='Batch size for fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=0.0005, help='Learning rate for fine-tuning')
    parser.add_argument('--ft_data_num', type=int, default=6, help='Numbers of fine-tuning samples')

    args = parser.parse_args()

    return args

def wsl_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','WSL')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

    setattr(args, 'pre_model_file', 'results/soh_pretraining_ev_data_results')

    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)

    ft_train = FtTrain(args)
    ft_train.train(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    tester = Test(args)
    true_capacity = test_y.copy()
    est_capacity, metric = tester.test(test_x.copy(), true_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def ssl_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','SSL')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    
    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)

    trainer = SSLTrain(args)
    trainer.ft_train(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    true_capacity = test_y.copy()
    est_capacity, metric = trainer.test(test_x.copy(), true_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def cnn_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','CNN')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    
    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)
    
    trainer = CNNTrain(args)
    trainer.train(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    true_capacity = test_y.copy()
    est_capacity, metric = trainer.test(test_x.copy(), true_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def benchmark_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','Benchmark')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    
    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)
    
    trainer = BenchmarkTrain(args)
    trainer.train(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    true_capacity = test_y.copy()
    est_capacity, metric = trainer.test(test_x.copy(), true_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def rf_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','RF')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    
    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)
    
    rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=9)
    rf.fit(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    true_capacity = test_y.copy()
    est_capacity = rf.predict(test_x)
    metric = eval_metrix(true_capacity, est_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def gpr_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','GPR')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    
    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)
    
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel,alpha=1e-5,n_restarts_optimizer=3,random_state=9)
    gpr.fit(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    true_capacity = test_y.copy()
    est_capacity = gpr.predict(test_x)
    metric = eval_metrix(true_capacity, est_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)

def svr_results(ft_x,ft_y,test_x,test_y,i):
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','SVR')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', target_dataset)
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    
    ft_files = os.path.join(args.save_folder, f'Experiment{i}')
    setattr(args, 'ft_files', ft_files)
    if not os.path.exists(args.ft_files):
        os.makedirs(args.ft_files)
    
    svr = SVR(kernel='rbf',epsilon=0.001)
    svr.fit(ft_x.copy(),ft_y.copy())

    eval_metrics = {}
    test_results = {}
    true_capacity = test_y.copy()
    est_capacity = svr.predict(test_x)
    metric = eval_metrix(true_capacity, est_capacity)
    
    test_results[f'field_data_est'] = est_capacity
    test_results[f'field_data_true'] = true_capacity
    eval_metrics['field_data'] = metric

    test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
    test_results_df.to_csv(f'{args.ft_files}/test_results.csv',index=False)
    
    eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE','MAE','MedAE','MAX'])
    eval_metrics_df.T.to_csv(f'{args.ft_files}/eval_metrics.csv', index=True)


def main():
    """
    Main function to run experiments
    运行每个数据集的主函数
    """

    # 在ev_data上预训练SSL模型
    args = get_args()
    target_dataset = args.target_dataset
    save_folder = os.path.join('results', 'soh_estimation_on_field_data','SSL')
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'source_dataset', 'Dataset 7')
    setattr(args, 'target_dataset', target_dataset)
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    print('Pre-training the model')
    pre_data = load_pre_data(args)
    setattr(args, 'ft_files', args.save_folder)
    trainer = SSLTrain(args)
    trainer.pre_train(pre_data['X'], pre_data['Y'])

    # 读取现场标签数据
    rd_labeled_data = np.load('data/Dataset 8/real_world_labeled_data.npz')
    X,Y = rd_labeled_data['Q_sequences'], rd_labeled_data['capacity']
    
    X = X/50
    Y = Y/155
    random.seed(15)
    for i in range(1,11):
        # 随机抽取6个样本用于微调、其余用于测试
        random_ids = random.sample(range(0, len(Y)), 6)
        ft_x,ft_y = X[random_ids],Y[random_ids]
        test_x = np.delete(X, random_ids, axis=0)
        test_y = np.delete(Y, random_ids, axis=0)

        wsl_results(ft_x,ft_y,test_x,test_y,i)
        ssl_results(ft_x,ft_y,test_x,test_y,i)
        cnn_results(ft_x,ft_y,test_x,test_y,i)
        benchmark_results(ft_x,ft_y,test_x,test_y,i)
        rf_results(ft_x,ft_y,test_x,test_y,i)
        gpr_results(ft_x,ft_y,test_x,test_y,i)
        svr_results(ft_x,ft_y,test_x,test_y,i)
        
if __name__ == '__main__':
    # main()
    pass
    