import torch
import numpy as np
from scipy.stats import spearmanr


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred_log, true_log, pred_raw=None, true_raw=None):
    mae_log = np.mean(np.abs(pred_log - true_log))
    rmse_log = np.sqrt(np.mean((pred_log - true_log) ** 2))
    
    if pred_raw is not None and true_raw is not None:
        mae_raw = np.mean(np.abs(pred_raw - true_raw))
        rmse_raw = np.sqrt(np.mean((pred_raw - true_raw) ** 2))
        spearman_corr, _ = spearmanr(pred_raw, true_raw)
        return {
            "mae_log": mae_log,
            "rmse_log": rmse_log,
            "mae_raw": mae_raw,
            "rmse_raw": rmse_raw,
            "spearman_corr": spearman_corr
        }
    
    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "spearman_corr": None
    }
