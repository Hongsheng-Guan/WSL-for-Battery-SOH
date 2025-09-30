from sklearn import metrics
import numpy as np

def eval_metrix(true_c, est_c):
    """
    Evaluate the metrics
    评估指标
    """
    RMSE = np.sqrt(metrics.mean_squared_error(true_c, est_c))
    MAPE = metrics.mean_absolute_percentage_error(true_c, est_c)
    MAE = metrics.mean_absolute_error(true_c, est_c)
    MedAE = metrics.median_absolute_error(true_c, est_c)
    MAX = metrics.max_error(true_c, est_c)
    
    return [RMSE, MAPE, MAE, MedAE, MAX]