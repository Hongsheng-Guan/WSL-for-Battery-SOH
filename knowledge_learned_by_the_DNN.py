import numpy as np
import torch
import os
import warnings
import seaborn as sns
import pandas as pd
import scienceplots
import umap
from models.CNN_BiLSTM import CNN_BiLSTM
from data_process.data_load import load_pre_data,load_test_data
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler

def extract_features(model, X):
    input_feat,cnn_feat,rec_feat,fnn_feat = [],[],[],[]
    with torch.no_grad():
        for xi in X:
            xi = torch.tensor(xi, dtype=torch.float)
            xi = xi.view(1, 50, 1)
            input_feat.append(xi.reshape(1,-1))

            xi = xi.permute(0,2,1)
            xi = model.cnn1(xi)
            xi = model.cnn2(xi)
            xi = xi.permute(0,2,1)
            cnn_feat.append(xi.reshape(1,-1))

            xi,_ = model.lstm1_1(xi)
            xi = model.lstm1_2(xi)
            xi,_ = model.lstm2_1(xi)
            xi = model.lstm2_2(xi)
            rec_feat.append(xi.reshape(1,-1))

            xi = model.flt(xi)
            xi = model.fc1(xi)
            fnn_feat.append(xi.reshape(1,-1))
    input_feat = torch.cat(input_feat).numpy()
    cnn_feat = torch.cat(cnn_feat).numpy()
    rec_feat = torch.cat(rec_feat).numpy()
    fnn_feat = torch.cat(fnn_feat).numpy()
    return [input_feat, cnn_feat, rec_feat, fnn_feat]

def plot_embeddings(embeddings, labels,op_conditions,save_path):
    plt.style.use('science')
    plt.style.use('ieee')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 6})
    fig, ax = plt.subplots(figsize=(55/25.4, 55/25.4))
    marks = ['o', '*', 's', '^', 'D', 'x']
    conditions = np.unique(op_conditions)
    for i,domain in enumerate(conditions):
        idx = op_conditions == domain
        scatter = sns.scatterplot(x=embeddings[idx, 0][::5], y=embeddings[idx, 1][::5],
                                hue=labels[idx][::5], palette='coolwarm',
                                marker=marks[i], 
                                edgecolor='black', s=10, linewidth=0.2)

    ax.set_xticks([0,10],[])
    ax.set_yticks([-5,5,15],[])
    ax.set_xlabel('Dimension 1', fontsize=6)
    ax.set_ylabel('Dimension 2', fontsize=6)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_embeddings_colorbar(embeddings, labels,op_conditions,save_path):
    plt.style.use('science')
    plt.style.use('ieee')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 6})
    fig, ax = plt.subplots(figsize=(55/25.4, 55/25.4))
    marks = ['o', '*', 's', '^', 'D', 'x']
    conditions = np.unique(op_conditions)
    for i,domain in enumerate(conditions):
        idx = op_conditions == domain
        scatter = sns.scatterplot(x=embeddings[idx, 0][::5], y=embeddings[idx, 1][::5], 
                                hue=labels[idx][::5], palette='coolwarm',
                                marker=marks[i], 
                                edgecolor='black', s=10, linewidth=0.2)
    norm = plt.Normalize(labels.min(), labels.max())
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    plt.colorbar(sm,ax=ax, label='SOH')  # add colorbar
    ax.set_xticks([0,10],[])
    ax.set_yticks([-5,5,15],[])
    ax.set_xlabel('Dimension 1', fontsize=6)
    ax.set_ylabel('Dimension 2', fontsize=6)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def stratified_sampling(domain_labels,samples_per_domain=1500, random_state=42):
    """
    对每个域进行分层采样，确保每个域有相同数量的样本
    """
    rng = np.random.default_rng(seed=random_state)

    unique_domains = np.unique(domain_labels)
    sampled_indices = []
    for domain in unique_domains:
        domain_indices = np.where(domain_labels == domain)[0]
        if len(domain_indices) <= samples_per_domain:
            selected_indices = domain_indices
        else:
            selected_indices = rng.choice(domain_indices, size=samples_per_domain, replace=False)
        sampled_indices.extend(selected_indices)
    
    return sampled_indices

def umap_plot():
    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3','Dataset 4', 'Dataset 5', 'Dataset 6']
    fig_name = ['figS16_d1', 'figS17_d2', 'figS18_d3', 'figS19_d4', 'figS20_d5', 'fig6']
    condition_code = {
            'Dataset 1': [1,2],
            'Dataset 2': [5,6,7],
            'Dataset 3': [8,9,10,11],
            'Dataset 4': [16,17,18],
            'Dataset 5': [19,20,21],
            'Dataset 6': [22,23,24,25,26,27]
        }
    ft_cells = ['Experiment1(25-1)','Experiment1(1C-4)','Experiment1(0-CC-2)','Experiment1(CY25-05_1-#14)','Experiment1(2C-6)','Experiment1(25_0.5b_100)']
    test_cells = {
        'Dataset 1':[['25-'+str(i) for i in range(3,9)],['45-'+str(i) for i in range(3,7)]],
        'Dataset 2':[['1C-'+str(i) for i in range(6,11)],['2C-4'],['3C-5','3C-6','3C-7','3C-9','3C-10']],
        'Dataset 3':[['0-CC-1','0-CC-3'],['10-CC-2','10-CC-3'],['25-CC-2','25-CC-3'],['40-CC-2','40-CC-3']],
        'Dataset 4':[['CY25-05_1-#12','CY25-05_1-#18','CY25-05_1-#19'],['CY35-05_1-#1'],['CY45-05_1-#21','CY45-05_1-#24','CY45-05_1-#25','CY45-05_1-#26','CY45-05_1-#27','CY45-05_1-#28']],
        'Dataset 5':[['2C-5','2C-8'],['3C-'+str(i) for i in range(7,16)],['4C-6']],
        'Dataset 6':[['25_0.5a_100'],['25_1b_100','25_1c_100','25_1d_100'],['25_2b_100'],['25_3a_100','25_3c_100','25_3d_100'],['35_1b_100','35_1c_100','35_1d_100'],['35_2a_100']],
    }
    
    for i in range(6):
        if i!=0:
            continue
        path_dir = 'figs/'+fig_name[i]
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        di = datasets[i]
        di_conditions = condition_code[di]
        di_test_cells = test_cells[di]
        x,sohs,conditions = [],[],[]
        for j in range(len(di_conditions)):
            for cell in di_test_cells[j]:
                data = load_test_data(di,cell)
                x.append(np.array(data['X']))
                sohs.append(np.array(data['Y']))
                conditions.append(np.array([di_conditions[j]]*len(data['Y'])))
        x = np.concatenate(x, axis=0)
        sohs = np.concatenate(sohs, axis=0)
        conditions = np.concatenate(conditions, axis=0)

        sampled_indices = stratified_sampling(conditions)
        x = x[sampled_indices]
        sohs = sohs[sampled_indices]
        conditions = conditions[sampled_indices]

        umap_model = umap.UMAP(
            n_neighbors=50,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )

        # 完成预训练的DNN的特征提取过程
        # Feature extraction process of the pre-trained DNN
        pre_path = os.path.join('results/soh_pretraining_ev_data_results','pre_trained_model.pth')
        pre_model = CNN_BiLSTM()
        pre_model.load_state_dict(torch.load(pre_path))
        pre_model.eval()
        pre_feas = extract_features(pre_model, x)
        for j in range(len(pre_feas)):
            pre_feaj_umap = umap_model.fit_transform(pre_feas[j])
            plot_embeddings(pre_feaj_umap, sohs, conditions, path_dir + f'/{di}_pre_{j+1}d.jpg')
            if j==3:
                plot_embeddings_colorbar(pre_feaj_umap, sohs, conditions, path_dir + f'/{di}_colorbar.jpg')
            
        # 微调DNN的特征提取过程
        # Feature extraction process of the fine-tuned DNN
        ft_path = os.path.join('results/soh_pretraining_ev_data_results',di,ft_cells[i],'from_Dataset 7_ft_model.pth')
        ft_model = CNN_BiLSTM()
        ft_model.load_state_dict(torch.load(ft_path))
        ft_model.eval()
        ft_feas = extract_features(ft_model, x)
        for j in range(len(ft_feas)):
            ft_feaj_umap = umap_model.fit_transform(ft_feas[j])
            plot_embeddings(ft_feaj_umap, sohs, conditions, path_dir + f'/{di}_ft_{j+1}d.jpg')

def compute_mmd(source_features, target_features, kernel='rbf', gamma=None):
    """
    计算源域和目标域特征之间的最大均值差异（MMD）
    
    参数:
    source_features: 源域特征，形状为 (n, 128) 的numpy数组或PyTorch张量
    target_features: 目标域特征，形状为 (m, 128) 的numpy数组或PyTorch张量
    kernel: 核函数类型，可选 'rbf'（径向基函数）或 'linear'（线性核）
    gamma: RBF核的带宽参数，如果为None则自动设置
    
    返回:
    mmd: MMD值
    """
    scaler = StandardScaler()
    all_features = np.vstack([source_features, target_features])
    scaler.fit(all_features)
    
    source_features = scaler.transform(source_features)
    target_features = scaler.transform(target_features)

    # 转换为PyTorch张量（如果输入是numpy数组）
    if isinstance(source_features, np.ndarray):
        source_features = torch.from_numpy(source_features)
    if isinstance(target_features, np.ndarray):
        target_features = torch.from_numpy(target_features)
    
    # 确保特征在同一设备上
    device = source_features.device
    target_features = target_features.to(device)
    
    n = source_features.size(0)
    m = target_features.size(0)
    
    if gamma is None:
        gamma = 1.0 / source_features.size(1)
    
    # 计算核矩阵
    if kernel == 'rbf':
        XX = torch.exp(-gamma * torch.cdist(source_features, source_features)**2)
        YY = torch.exp(-gamma * torch.cdist(target_features, target_features)**2)
        XY = torch.exp(-gamma * torch.cdist(source_features, target_features)**2)
    elif kernel == 'linear':
        XX = torch.mm(source_features, source_features.t())
        YY = torch.mm(target_features, target_features.t())
        XY = torch.mm(source_features, target_features.t())
    else:
        raise ValueError("不支持的核类型。请选择 'rbf' 或 'linear'")
    
    # 计算MMD（无偏估计）
    mmd = (XX.sum() / (n * (n - 1)) + 
           YY.sum() / (m * (m - 1)) - 
           2 * XY.sum() / (n * m))
    
    return mmd.item()
    

def get_mmd():
    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3','Dataset 4', 'Dataset 5', 'Dataset 6']
    condition_code = {
            'Dataset 1': [1,2],
            'Dataset 2': [5,6,7],
            'Dataset 3': [8,9,10,11],
            'Dataset 4': [16,17,18],
            'Dataset 5': [19,20,21],
            'Dataset 6': [22,23,24,25,26,27]
        }
    ft_cells = ['Experiment1(25-1)','Experiment1(1C-4)','Experiment1(0-CC-2)','Experiment1(CY25-05_1-#14)','Experiment1(2C-6)','Experiment1(25_0.5b_100)']
    source_conditions = [1,5,8,16,19,22]
    test_cells = {
        'Dataset 1':[['25-'+str(i) for i in range(3,9)],['45-'+str(i) for i in range(3,7)]],
        'Dataset 2':[['1C-'+str(i) for i in range(6,11)],['2C-4'],['3C-5','3C-6','3C-7','3C-9','3C-10']],
        'Dataset 3':[['0-CC-1','0-CC-3'],['10-CC-2','10-CC-3'],['25-CC-2','25-CC-3'],['40-CC-2','40-CC-3']],
        'Dataset 4':[['CY25-05_1-#12','CY25-05_1-#18','CY25-05_1-#19'],['CY35-05_1-#1'],['CY45-05_1-#21','CY45-05_1-#24','CY45-05_1-#25','CY45-05_1-#26','CY45-05_1-#27','CY45-05_1-#28']],
        'Dataset 5':[['2C-5','2C-8'],['3C-'+str(i) for i in range(7,16)],['4C-6']],
        'Dataset 6':[['25_0.5a_100'],['25_1b_100','25_1c_100','25_1d_100'],['25_2b_100'],['25_3a_100','25_3c_100','25_3d_100'],['35_1b_100','35_1c_100','35_1d_100'],['35_2a_100']],
    }
    mmd_results = {'task':[],'mmd_of_wsl':[],'mmd_of_benchmark':[]}
    soh_errors = {'task':[],'rmse_of_wsl':[],'rmse_of_benchmark':[]}
    for i in range(6):
        di = datasets[i]
        di_conditions = condition_code[di]
        di_test_cells = test_cells[di]
        source_condi = source_conditions[i]
        x,sohs,conditions = [],[],[]
        for j in range(len(di_conditions)):
            wsl_rmses = []
            bh_rmses = []
            for cell in di_test_cells[j]:
                data = load_test_data(di,cell)
                x.append(np.array(data['X']))
                sohs.append(np.array(data['Y']))
                conditions.append(np.array([di_conditions[j]]*len(data['Y'])))

                if source_condi != di_conditions[j]:
                    wsl_result_path = os.path.join('results/soh_pretraining_ev_data_results',di,ft_cells[i],'eval_metrics.csv')
                    wsl_metrics = pd.read_csv(wsl_result_path)
                    wsl_cell_row = wsl_metrics[wsl_metrics.iloc[:, 0] == cell]
                    wsl_rmses.append(wsl_cell_row['RMSE'].values[0])

                    bh_result_path = os.path.join('results/comparison_methods_with_limited_labels_results/Benchmark',di,ft_cells[i],'eval_metrics.csv')
                    bh_metrics = pd.read_csv(bh_result_path)
                    bh_cell_row = bh_metrics[bh_metrics.iloc[:, 0] == cell]
                    bh_rmses.append(bh_cell_row['RMSE'].values[0])

            if source_condi != di_conditions[j]:
                soh_errors['task'].append(f'#{source_condi} to #{di_conditions[j]}')
                soh_errors['rmse_of_wsl'].append(f'{np.mean(wsl_rmses)*100:.3f}')
                soh_errors['rmse_of_benchmark'].append(f'{np.mean(bh_rmses)*100:.3f}')

        x = np.concatenate(x, axis=0)
        sohs = np.concatenate(sohs, axis=0)
        conditions = np.concatenate(conditions, axis=0)
        
        # 微调DNN的特征提取过程
        # Feature extraction process of the fine-tuned DNN
        ft_path = os.path.join('results/soh_pretraining_ev_data_results',di,ft_cells[i],'from_Dataset 7_ft_model.pth')
        ft_model = CNN_BiLSTM()
        ft_model.load_state_dict(torch.load(ft_path))
        ft_model.eval()
        ft_feas = extract_features(ft_model, x)[-1]

        # Benchmark的特征提取过程
        # Feature extraction process of the Benchmark
        bh_path = os.path.join('results/comparison_methods_with_limited_labels_results/Benchmark',di,ft_cells[i],'Benchmark_model.pth')
        bh_model = CNN_BiLSTM()
        bh_model.load_state_dict(torch.load(bh_path))
        bh_model.eval()
        bh_feas = extract_features(bh_model, x)[-1]
        print(di)
        
        source_feat = ft_feas[conditions == source_condi]
        for target_condi in di_conditions:
            if source_condi == target_condi:
                continue
            target_feat = ft_feas[conditions == target_condi]

            mmd_results['task'].append(f'#{source_condi} to #{target_condi}')
            mmd = compute_mmd(source_feat,target_feat)
            mmd_results['mmd_of_wsl'].append(f'{mmd:.5f}')
            
        source_feat = bh_feas[conditions == source_condi]
        for target_condi in di_conditions:
            if source_condi == target_condi:
                continue
            target_feat = bh_feas[conditions == target_condi]
            mmd = compute_mmd(source_feat,target_feat)
            mmd_results['mmd_of_benchmark'].append(f'{mmd:.5f}')

    mmd_df = pd.DataFrame(mmd_results)
    mmd_df.to_csv('figs/fig6/mmd_result_to_fc1.csv',index=False)
    rmse_df = pd.DataFrame(soh_errors)
    rmse_df.to_csv('figs/fig6/soh_errors.csv',index=False)

if __name__ == "__main__":
    # path_dir = 'figs/fig6'
    # if not os.path.exists(path_dir):
    #     os.makedirs(path_dir)
    # get_mmd()
    # umap_plot()
    pass

                
            

