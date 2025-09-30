import os
import pandas as pd
import numpy as np

def get_rmse(path,dataset):
    intra_condition_rmses,cross_condition_rmses = [],[]
    cell_df = pd.read_csv(os.path.join('data', dataset, 'train_cells_id.csv'))
    ft_cells = cell_df['ft_cells'].values
    for i,ft_cell in enumerate(ft_cells):
        if str(ft_cell)=='nan':
            continue
        df = pd.read_csv(path+'/Experiment'+str(i+1)+'('+ft_cell+')/eval_metrics.csv')
        for j in range(len(df)):
            test_cell = df.iloc[j,0]
            if dataset in ['Dataset 1','Dataset 2','Dataset 5']: # The condition number for datasets 1,2 and 5 is the first 2 characters
                if ft_cell[:2] == test_cell[:2]:
                    intra_condition_rmses.append(df.iloc[j]['RMSE']*100)
                else:
                    cross_condition_rmses.append(df.iloc[j]['RMSE']*100)
            elif dataset in ['Dataset 3','Dataset 4','Dataset 6']: # The condition number for datasets 3,4 and 6 is the first 4 characters
                if ft_cell[:4] == test_cell[:4]:
                    intra_condition_rmses.append(df.iloc[j]['RMSE']*100)
                else:
                    cross_condition_rmses.append(df.iloc[j]['RMSE']*100)
    return intra_condition_rmses,cross_condition_rmses

def get_mape(path,dataset):
    intra_condition_mapes,cross_condition_mapes = [],[]
    cell_df = pd.read_csv(os.path.join('data', dataset, 'train_cells_id.csv'))
    ft_cells = cell_df['ft_cells'].values
    for i,ft_cell in enumerate(ft_cells):
        if str(ft_cell)=='nan':
            continue
        df = pd.read_csv(path+'/Experiment'+str(i+1)+'('+ft_cell+')/eval_metrics.csv')
        for j in range(len(df)):
            test_cell = df.iloc[j,0]
            if dataset in ['Dataset 1','Dataset 2','Dataset 5']: # The condition number for datasets 1,2 and 5 is the first 2 characters
                if ft_cell[:2] == test_cell[:2]:
                    intra_condition_mapes.append(df.iloc[j]['MAPE']*100)
                else:
                    cross_condition_mapes.append(df.iloc[j]['MAPE']*100)
            elif dataset in ['Dataset 3','Dataset 4','Dataset 6']: # The condition number for datasets 3,4 and 6 is the first 4 characters
                if ft_cell[:4] == test_cell[:4]:
                    intra_condition_mapes.append(df.iloc[j]['MAPE']*100)
                else:
                    cross_condition_mapes.append(df.iloc[j]['MAPE']*100)
    return intra_condition_mapes,cross_condition_mapes

def get_mae(path,dataset):
    intra_condition_maes,cross_condition_maes = [],[]
    cell_df = pd.read_csv(os.path.join('data', dataset, 'train_cells_id.csv'))
    ft_cells = cell_df['ft_cells'].values
    for i,ft_cell in enumerate(ft_cells):
        if str(ft_cell)=='nan':
            continue
        df = pd.read_csv(path+'/Experiment'+str(i+1)+'('+ft_cell+')/eval_metrics.csv')
        for j in range(len(df)):
            test_cell = df.iloc[j,0]
            if dataset in ['Dataset 1','Dataset 2','Dataset 5']: # The condition number for datasets 1,2 and 5 is the first 2 characters
                if ft_cell[:2] == test_cell[:2]:
                    intra_condition_maes.append(df.iloc[j]['MAE']*100)
                else:
                    cross_condition_maes.append(df.iloc[j]['MAE']*100)
            elif dataset in ['Dataset 3','Dataset 4','Dataset 6']: # The condition number for datasets 3,4 and 6 is the first 4 characters
                if ft_cell[:4] == test_cell[:4]:
                    intra_condition_maes.append(df.iloc[j]['MAE']*100)
                else:
                    cross_condition_maes.append(df.iloc[j]['MAE']*100)
    return intra_condition_maes,cross_condition_maes

def get_medae(path,dataset):
    intra_condition_medaes,cross_condition_medaes = [],[]
    cell_df = pd.read_csv(os.path.join('data', dataset, 'train_cells_id.csv'))
    ft_cells = cell_df['ft_cells'].values
    for i,ft_cell in enumerate(ft_cells):
        if str(ft_cell)=='nan':
            continue
        df = pd.read_csv(path+'/Experiment'+str(i+1)+'('+ft_cell+')/eval_metrics.csv')
        for j in range(len(df)):
            test_cell = df.iloc[j,0]
            if dataset in ['Dataset 1','Dataset 2','Dataset 5']: # The condition number for datasets 1,2 and 5 is the first 2 characters
                if ft_cell[:2] == test_cell[:2]:
                    intra_condition_medaes.append(df.iloc[j]['MedAE']*100)
                else:
                    cross_condition_medaes.append(df.iloc[j]['MedAE']*100)
            elif dataset in ['Dataset 3','Dataset 4','Dataset 6']: # The condition number for datasets 3,4 and 6 is the first 4 characters
                if ft_cell[:4] == test_cell[:4]:
                    intra_condition_medaes.append(df.iloc[j]['MedAE']*100)
                else:
                    cross_condition_medaes.append(df.iloc[j]['MedAE']*100)
    return intra_condition_medaes,cross_condition_medaes

def get_max(path,dataset):
    intra_condition_maxs,cross_condition_maxs = [],[]
    cell_df = pd.read_csv(os.path.join('data', dataset, 'train_cells_id.csv'))
    ft_cells = cell_df['ft_cells'].values
    for i,ft_cell in enumerate(ft_cells):
        if str(ft_cell)=='nan':
            continue
        df = pd.read_csv(path+'/Experiment'+str(i+1)+'('+ft_cell+')/eval_metrics.csv')
        for j in range(len(df)):
            test_cell = df.iloc[j,0]
            if dataset in ['Dataset 1','Dataset 2','Dataset 5']: # The condition number for datasets 1,2 and 5 is the first 2 characters
                if ft_cell[:2] == test_cell[:2]:
                    intra_condition_maxs.append(df.iloc[j]['MAX']*100)
                else:
                    cross_condition_maxs.append(df.iloc[j]['MAX']*100)
            elif dataset in ['Dataset 3','Dataset 4','Dataset 6']: # The condition number for datasets 3,4 and 6 is the first 4 characters
                if ft_cell[:4] == test_cell[:4]:
                    intra_condition_maxs.append(df.iloc[j]['MAX']*100)
                else:
                    cross_condition_maxs.append(df.iloc[j]['MAX']*100)
    return intra_condition_maxs,cross_condition_maxs

def get_results(path,dataset):
    intra_condition_estimates,intra_condition_ture = [],[]
    cross_condition_estimates,cross_condition_ture = [],[]
    cell_df = pd.read_csv(os.path.join('data', dataset, 'train_cells_id.csv'))
    ft_cells = cell_df['ft_cells'].values
    for i,ft_cell in enumerate(ft_cells):
        if str(ft_cell)=='nan':
            continue
        df = pd.read_csv(path+'/Experiment'+str(i+1)+'('+ft_cell+')/test_results.csv')

        keys = df.keys()
        test_cells = []
        for ki in keys:
            if 'est' in ki:
                ki = ki[:-4]
            elif 'true' in ki:
                ki = ki[:-5]
            if ki not in test_cells:
                test_cells.append(ki)
        
        for cell in test_cells:
            if dataset in ['Dataset 1','Dataset 2','Dataset 5']:# The condition number for datasets 1,2 and 5 is the first 2 characters
                if ft_cell[:2] == cell[:2]:
                    intra_esti = df[cell+'_est'].values
                    intra_esti = intra_esti[~np.isnan(intra_esti)]
                    intra_true = df[cell+'_true'].values
                    intra_true = intra_true[~np.isnan(intra_true)]
                    intra_condition_estimates.append(intra_esti)
                    intra_condition_ture.append(intra_true)
                else:
                    cross_esti = df[cell+'_est'].values
                    cross_esti = cross_esti[~np.isnan(cross_esti)]
                    cross_true = df[cell+'_true'].values
                    cross_true = cross_true[~np.isnan(cross_true)]
                    cross_condition_estimates.append(cross_esti)
                    cross_condition_ture.append(cross_true)
            elif dataset in ['Dataset 3','Dataset 4','Dataset 6']: # The condition number for datasets 3,4 and 6 is the first 4 characters
                if ft_cell[:4] == cell[:4]:
                    intra_esti = df[cell+'_est'].values
                    intra_esti = intra_esti[~np.isnan(intra_esti)]
                    intra_true = df[cell+'_true'].values
                    intra_true = intra_true[~np.isnan(intra_true)]
                    intra_condition_estimates.append(intra_esti)
                    intra_condition_ture.append(intra_true)
                else:
                    cross_esti = df[cell+'_est'].values
                    cross_esti = cross_esti[~np.isnan(cross_esti)]
                    cross_true = df[cell+'_true'].values
                    cross_true = cross_true[~np.isnan(cross_true)]
                    cross_condition_estimates.append(cross_esti)
                    cross_condition_ture.append(cross_true)

    intra_condition_estimates = np.concatenate(intra_condition_estimates)
    intra_condition_ture = np.concatenate(intra_condition_ture)
    cross_condition_estimates = np.concatenate(cross_condition_estimates)
    cross_condition_ture = np.concatenate(cross_condition_ture)

    return intra_condition_estimates,intra_condition_ture,cross_condition_estimates,cross_condition_ture