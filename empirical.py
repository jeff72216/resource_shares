import os
import torch
import gc
import scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from multiprocessing import Pool, shared_memory
from tqdm import tqdm

# Define functions --------------------
def mle(params, q_batch_f, q_batch_m, q_batch_c, 
        pq_batch_f, pq_batch_m, pq_batch_c, 
        tp_batch_f, tp_batch_m, tp_batch_c, 
        log_w_batch_f, log_w_batch_m, log_w_batch_c,
        delta_batch_f, delta_batch_m, delta_batch_c,
        trend_batch_f, trend_batch_m, trend_batch_c,
        yr_batch_f, yr_batch_m, yr_batch_c,
        param_idx, n_attr_f, n_attr_m, n_attr_c,
        n_upc_f, n_upc_m, n_upc_c, delta_idx_2004, device):
    gama_f = params[
        :, range(param_idx[0], param_idx[1])].view(n_attr_f, n_upc_f)
    gama_m = params[
        :, range(param_idx[1], param_idx[2])].view(n_attr_m, n_upc_m)
    gama_c = params[
        :, range(param_idx[2], param_idx[3])].view(n_attr_c, n_upc_c)
    alfa_f = torch.stack([
        params[:, range(param_idx[3], param_idx[4])].squeeze(),
        params[:, range(param_idx[5], param_idx[6])].squeeze(),
        params[:, range(param_idx[8], param_idx[9])].squeeze()
    ])[tp_batch_f]
    alfa_m = torch.stack([
        params[:, range(param_idx[4], param_idx[5])].squeeze(),
        params[:, range(param_idx[6], param_idx[7])].squeeze(),
        params[:, range(param_idx[9], param_idx[10])].squeeze()
    ])[tp_batch_m]
    alfa_c = torch.stack([
        params[:, range(param_idx[7], param_idx[8])].squeeze(),
        params[:, range(param_idx[10], param_idx[11])].squeeze()
    ])[tp_batch_c-1]
    delta = params[:, range(param_idx[11], param_idx[12])].squeeze()
    delta_f = delta[tp_batch_f]
    delta_m = delta[tp_batch_m+3]
    delta_c = delta[tp_batch_c+5]
    beta2 = torch.exp(params[:, range(param_idx[12], param_idx[13])].squeeze())
    beta2_f = beta2[tp_batch_f].unsqueeze(1)
    beta2_m = beta2[tp_batch_m].unsqueeze(1)
    beta2_c = beta2[tp_batch_c].unsqueeze(1)
    lambda_w = params[:, range(param_idx[13], param_idx[14])].squeeze()
    lambda_w_f = lambda_w[tp_batch_f]
    lambda_w_m = lambda_w[tp_batch_m+3]
    lambda_w_c = lambda_w[tp_batch_c+5]
    mask_f = torch.isin(delta_batch_f, delta_idx_2004)
    mask_m = torch.isin(delta_batch_m, delta_idx_2004)
    mask_c = torch.isin(delta_batch_c, delta_idx_2004)
    delta_batch_f_new = delta_batch_f - 3
    delta_batch_m_new = delta_batch_m - 6
    delta_batch_c_new = delta_batch_c - 8
    lambda_yr = params[:, range(param_idx[14], param_idx[15])].squeeze()
    lambda_yr_f = torch.zeros_like(lambda_w_f, device=device)
    lambda_yr_f[~mask_f] = lambda_yr[delta_batch_f_new[~mask_f]]
    lambda_yr_m = torch.zeros_like(lambda_w_m, device=device)
    lambda_yr_m[~mask_m] = lambda_yr[delta_batch_m_new[~mask_m]]
    lambda_yr_c = torch.zeros_like(lambda_w_c, device=device)
    lambda_yr_c[~mask_c] = lambda_yr[delta_batch_c_new[~mask_c]]
    lambda_yr_w = params[:, range(param_idx[15], param_idx[16])].squeeze()
    lambda_yr_w_f = lambda_yr_w[tp_batch_f]
    lambda_yr_w_m = lambda_yr_w[tp_batch_m+3]
    lambda_yr_w_c = lambda_yr_w[tp_batch_c+5]
    alfa_f[~mask_f] += params[:, range(param_idx[16], param_idx[17])].view(
        16, n_attr_f)[yr_batch_f[~mask_f]-1]
    alfa_m[~mask_m] += params[:, range(param_idx[17], param_idx[18])].view(
        16, n_attr_m)[yr_batch_m[~mask_m]-1]
    alfa_c[~mask_c] += params[:, range(param_idx[18], param_idx[19])].view(
        16, n_attr_c)[yr_batch_c[~mask_c]-1]
    n_obs_f = len(tp_batch_f)
    n_obs_m = len(tp_batch_m)
    n_obs_c = len(tp_batch_c)

    delta_f = torch.exp(
        delta_f + lambda_yr_f + lambda_w_f*log_w_batch_f 
        + lambda_yr_w_f*log_w_batch_f*trend_batch_f).unsqueeze(1)
    delta_m = torch.exp(
        delta_m + lambda_yr_m + lambda_w_m*log_w_batch_m 
        + lambda_yr_w_m*log_w_batch_m*trend_batch_m).unsqueeze(1)
    delta_c = torch.exp(
        delta_c + lambda_yr_c + lambda_w_c*log_w_batch_c 
        + lambda_yr_w_c*log_w_batch_c*trend_batch_c).unsqueeze(1)

    attr_f = torch.matmul(q_batch_f.to_dense().transpose(1, 2), gama_f.T)
    attr_m = torch.matmul(q_batch_m.to_dense().transpose(1, 2), gama_m.T)
    attr_c = torch.matmul(q_batch_c.to_dense().transpose(1, 2), gama_c.T)

    vi_1_f = torch.bmm(attr_f, alfa_f.unsqueeze(2)).squeeze(2)
    vi_1_m = torch.bmm(attr_m, alfa_m.unsqueeze(2)).squeeze(2)
    vi_1_c = torch.bmm(attr_c, alfa_c.unsqueeze(2)).squeeze(2)

    vi_2_f = torch.matmul(
        attr_f.permute(1, 0, 2), attr_f.permute(1, 2, 0)
    ).diagonal(dim1=1, dim2=2).transpose(1, 0)
    vi_2_m = torch.matmul(
        attr_m.permute(1, 0, 2), attr_m.permute(1, 2, 0)
    ).diagonal(dim1=1, dim2=2).transpose(1, 0)
    vi_2_c = torch.matmul(
        attr_c.permute(1, 0, 2), attr_c.permute(1, 2, 0)
    ).diagonal(dim1=1, dim2=2).transpose(1, 0)

    vi_3_f = delta_f * pq_batch_f
    vi_3_m = delta_m * pq_batch_m
    vi_3_c = delta_c * pq_batch_c

    vi_4_f = beta2_f * pq_batch_f * pq_batch_f
    vi_4_m = beta2_m * pq_batch_m * pq_batch_m
    vi_4_c = beta2_c * pq_batch_c * pq_batch_c

    vi_f = vi_1_f - vi_2_f - vi_3_f - vi_4_f
    vi_m = vi_1_m - vi_2_m - vi_3_m - vi_4_m
    vi_c = vi_1_c - vi_2_c - vi_3_c - vi_4_c

    # Numerically stable logit functions
    vi_max_f = torch.max(vi_f, axis=1)[0]
    vi_max_m = torch.max(vi_m, axis=1)[0]
    vi_max_c = torch.max(vi_c, axis=1)[0]

    logit_f_1 = vi_f[:, 0] - vi_max_f
    logit_m_1 = vi_m[:, 0] - vi_max_m
    logit_c_1 = vi_c[:, 0] - vi_max_c

    logit_f_2 = torch.sum(
        torch.exp(vi_f - vi_max_f.view(n_obs_f, 1)), dim=1)
    logit_m_2 = torch.sum(
        torch.exp(vi_m - vi_max_m.view(n_obs_m, 1)), dim=1)
    logit_c_2 = torch.sum(
        torch.exp(vi_c - vi_max_c.view(n_obs_c, 1)), dim=1)

    logit_f = torch.sum(logit_f_1 - torch.log(logit_f_2))
    logit_m = torch.sum(logit_m_1 - torch.log(logit_m_2))
    logit_c = torch.sum(logit_c_1 - torch.log(logit_c_2))

    return -((logit_f+logit_m+logit_c) / (n_obs_f+n_obs_m+n_obs_c))

def bootstrap(args):
    # Pass arguments
    (
        idx_bt, init, param_idx, delta_idx_2004,
        price_f, price_m, price_c,
        candidate_f, candidate_m, candidate_c,
        bt_idx_f, bt_idx_m, bt_idx_c, 
        nobs_f, nobs_m, nobs_c, 
        n_upc_f, n_upc_m, n_upc_c, 
        len_train_f, len_train_m, len_train_c,
        len_valdn_f, len_valdn_m, len_valdn_c,
        len_val_f, len_val_m, len_val_c,
        len_valdn_val_f, len_valdn_val_m, len_valdn_val_c,
        row_f_shm_name, row_m_shm_name, row_c_shm_name, 
        col_f_shm_name, col_m_shm_name, col_c_shm_name, 
        q_val_f_shm_name, q_val_m_shm_name, q_val_c_shm_name, 
        p_val_f_shm_name, p_val_m_shm_name, p_val_c_shm_name, 
        tp_idx_f_shm_name, tp_idx_m_shm_name, tp_idx_c_shm_name, 
        delta_idx_f_shm_name, delta_idx_m_shm_name, delta_idx_c_shm_name, 
        log_w_f_shm_name, log_w_m_shm_name, log_w_c_shm_name,
        trend_f_shm_name, trend_m_shm_name, trend_c_shm_name, 
        yr_idx_f_shm_name, yr_idx_m_shm_name, yr_idx_c_shm_name,
        q_valdn_coo_f_shm_name, q_valdn_coo_m_shm_name, q_valdn_coo_c_shm_name, 
        q_valdn_val_f_shm_name, q_valdn_val_m_shm_name, q_valdn_val_c_shm_name, 
        pq_valdn_f_shm_name, pq_valdn_m_shm_name, pq_valdn_c_shm_name, 
        tp_valdn_f_shm_name, tp_valdn_m_shm_name, tp_valdn_c_shm_name, 
        log_w_valdn_f_shm_name, log_w_valdn_m_shm_name, log_w_valdn_c_shm_name,
        trend_valdn_f_shm_name, trend_valdn_m_shm_name, trend_valdn_c_shm_name, 
        delta_valdn_f_shm_name, delta_valdn_m_shm_name, delta_valdn_c_shm_name, 
        yr_valdn_f_shm_name, yr_valdn_m_shm_name, yr_valdn_c_shm_name, 
        batch_size_f, batch_size_m, batch_size_c, seed
    ) = args

    # Parameters
    n_attr_f = 5
    n_attr_m = 5
    n_attr_c = 5
    n_imposters = 50
    max_q = 3
    device = torch.device("cpu")
    torch.manual_seed(seed)
    n_batch = len(bt_idx_f)

    # Hyperparameters
    lr = 1e-3
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    patience = 30
    max_epoch = 700

    # Access shared memory
    row_f_shm = shared_memory.SharedMemory(name=row_f_shm_name)
    shared_row_f = np.ndarray(
        (len_val_f, ), dtype=np.int16, buffer=row_f_shm.buf)
    row_f = torch.from_numpy(shared_row_f.astype(np.int32))
    col_f_shm = shared_memory.SharedMemory(name=col_f_shm_name)
    shared_col_f = np.ndarray(
        (len_val_f, ), dtype=np.int16, buffer=col_f_shm.buf)
    col_f = torch.from_numpy(shared_col_f.astype(np.int32))
    q_val_f_shm = shared_memory.SharedMemory(name=q_val_f_shm_name)
    shared_q_val_f = np.ndarray(
        (len_val_f, ), dtype=np.int8, buffer=q_val_f_shm.buf)
    q_val_f = torch.from_numpy(shared_q_val_f.astype(np.float32))
    p_val_f_shm = shared_memory.SharedMemory(name=p_val_f_shm_name)
    shared_p_val_f = np.ndarray(
        (len_val_f, ), dtype=np.float32, buffer=p_val_f_shm.buf)
    p_val_f = torch.from_numpy(shared_p_val_f.astype(np.float32))
    tp_idx_f_shm = shared_memory.SharedMemory(name=tp_idx_f_shm_name)
    shared_tp_idx_f = np.ndarray(
        (nobs_f, ), dtype=np.int8, buffer=tp_idx_f_shm.buf)
    tp_idx_f = torch.from_numpy(shared_tp_idx_f.astype(np.int32))
    delta_idx_f_shm = shared_memory.SharedMemory(name=delta_idx_f_shm_name)
    shared_delta_idx_f = np.ndarray(
        (nobs_f, ), dtype=np.int16, buffer=delta_idx_f_shm.buf)
    delta_idx_f = torch.from_numpy(shared_delta_idx_f.astype(np.int32))
    log_w_f_shm = shared_memory.SharedMemory(name=log_w_f_shm_name)
    shared_log_w_f = np.ndarray(
        (nobs_f, ), dtype=np.float32, buffer=log_w_f_shm.buf)
    log_w_f = torch.from_numpy(shared_log_w_f)
    trend_f_shm = shared_memory.SharedMemory(name=trend_f_shm_name)
    shared_trend_f = np.ndarray(
        (nobs_f, ), dtype=np.float32, buffer=trend_f_shm.buf)
    trend_f = torch.from_numpy(shared_trend_f)
    yr_idx_f_shm = shared_memory.SharedMemory(name=yr_idx_f_shm_name)
    yr_idx_f = np.ndarray(
        (nobs_f, ), dtype=np.int8, buffer=yr_idx_f_shm.buf)
    
    row_m_shm = shared_memory.SharedMemory(name=row_m_shm_name)
    shared_row_m = np.ndarray(
        (len_val_m, ), dtype=np.int16, buffer=row_m_shm.buf)
    row_m = torch.from_numpy(shared_row_m.astype(np.int32))
    col_m_shm = shared_memory.SharedMemory(name=col_m_shm_name)
    shared_col_m = np.ndarray(
        (len_val_m, ), dtype=np.int16, buffer=col_m_shm.buf)
    col_m = torch.from_numpy(shared_col_m.astype(np.int32))
    q_val_m_shm = shared_memory.SharedMemory(name=q_val_m_shm_name)
    shared_q_val_m = np.ndarray(
        (len_val_m, ), dtype=np.int8, buffer=q_val_m_shm.buf)
    q_val_m = torch.from_numpy(shared_q_val_m.astype(np.float32))
    p_val_m_shm = shared_memory.SharedMemory(name=p_val_m_shm_name)
    shared_p_val_m = np.ndarray(
        (len_val_m, ), dtype=np.float32, buffer=p_val_m_shm.buf)
    p_val_m = torch.from_numpy(shared_p_val_m.astype(np.float32))
    tp_idx_m_shm = shared_memory.SharedMemory(name=tp_idx_m_shm_name)
    shared_tp_idx_m = np.ndarray(
        (nobs_m, ), dtype=np.int8, buffer=tp_idx_m_shm.buf)
    tp_idx_m = torch.from_numpy(shared_tp_idx_m.astype(np.int32))
    delta_idx_m_shm = shared_memory.SharedMemory(name=delta_idx_m_shm_name)
    shared_delta_idx_m = np.ndarray(
        (nobs_m, ), dtype=np.int16, buffer=delta_idx_m_shm.buf)
    delta_idx_m = torch.from_numpy(shared_delta_idx_m.astype(np.int32))
    log_w_m_shm = shared_memory.SharedMemory(name=log_w_m_shm_name)
    shared_log_w_m = np.ndarray(
        (nobs_m, ), dtype=np.float32, buffer=log_w_m_shm.buf)
    log_w_m = torch.from_numpy(shared_log_w_m)
    trend_m_shm = shared_memory.SharedMemory(name=trend_m_shm_name)
    shared_trend_m = np.ndarray(
        (nobs_m, ), dtype=np.float32, buffer=trend_m_shm.buf)
    trend_m = torch.from_numpy(shared_trend_m)
    yr_idx_m_shm = shared_memory.SharedMemory(name=yr_idx_m_shm_name)
    yr_idx_m = np.ndarray(
        (nobs_m, ), dtype=np.int8, buffer=yr_idx_m_shm.buf)

    row_c_shm = shared_memory.SharedMemory(name=row_c_shm_name)
    shared_row_c = np.ndarray(
        (len_val_c, ), dtype=np.int16, buffer=row_c_shm.buf)
    row_c = torch.from_numpy(shared_row_c.astype(np.int32))
    col_c_shm = shared_memory.SharedMemory(name=col_c_shm_name)
    shared_col_c = np.ndarray(
        (len_val_c, ), dtype=np.int16, buffer=col_c_shm.buf)
    col_c = torch.from_numpy(shared_col_c.astype(np.int32))
    q_val_c_shm = shared_memory.SharedMemory(name=q_val_c_shm_name)
    shared_q_val_c = np.ndarray(
        (len_val_c, ), dtype=np.int8, buffer=q_val_c_shm.buf)
    q_val_c = torch.from_numpy(shared_q_val_c.astype(np.float32))
    p_val_c_shm = shared_memory.SharedMemory(name=p_val_c_shm_name)
    shared_p_val_c = np.ndarray(
        (len_val_c, ), dtype=np.float32, buffer=p_val_c_shm.buf)
    p_val_c = torch.from_numpy(shared_p_val_c.astype(np.float32))
    tp_idx_c_shm = shared_memory.SharedMemory(name=tp_idx_c_shm_name)
    shared_tp_idx_c = np.ndarray(
        (nobs_c, ), dtype=np.int8, buffer=tp_idx_c_shm.buf)
    tp_idx_c = torch.from_numpy(shared_tp_idx_c.astype(np.int32))
    delta_idx_c_shm = shared_memory.SharedMemory(name=delta_idx_c_shm_name)
    shared_delta_idx_c = np.ndarray(
        (nobs_c, ), dtype=np.int16, buffer=delta_idx_c_shm.buf)
    delta_idx_c = torch.from_numpy(shared_delta_idx_c.astype(np.int32))
    log_w_c_shm = shared_memory.SharedMemory(name=log_w_c_shm_name)
    shared_log_w_c = np.ndarray(
        (nobs_c, ), dtype=np.float32, buffer=log_w_c_shm.buf)
    log_w_c = torch.from_numpy(shared_log_w_c)
    trend_c_shm = shared_memory.SharedMemory(name=trend_c_shm_name)
    shared_trend_c = np.ndarray(
        (nobs_c, ), dtype=np.float32, buffer=trend_c_shm.buf)
    trend_c = torch.from_numpy(shared_trend_c)
    yr_idx_c_shm = shared_memory.SharedMemory(name=yr_idx_c_shm_name)
    yr_idx_c = np.ndarray(
        (nobs_c, ), dtype=np.int8, buffer=yr_idx_c_shm.buf)

    q_valdn_coo_f_shm = shared_memory.SharedMemory(name=q_valdn_coo_f_shm_name)
    shared_q_valdn_coo_f = np.ndarray(
        (3, len_valdn_val_f), 
        dtype=np.int16, buffer=q_valdn_coo_f_shm.buf)
    q_valdn_coo_f = torch.from_numpy(shared_q_valdn_coo_f.astype(np.int32))
    q_valdn_val_f_shm = shared_memory.SharedMemory(name=q_valdn_val_f_shm_name)
    shared_q_valdn_val_f = np.ndarray(
        (len_valdn_val_f, ), 
        dtype=np.int16, buffer=q_valdn_val_f_shm.buf)
    q_valdn_val_f = torch.from_numpy(shared_q_valdn_val_f.astype(np.float32))
    q_valdn_f = torch.sparse_coo_tensor(
        indices=q_valdn_coo_f,
        values=q_valdn_val_f,
        size=(len_valdn_f, n_upc_f, n_imposters+1)
    )
    pq_valdn_f_shm = shared_memory.SharedMemory(name=pq_valdn_f_shm_name)
    shared_pq_valdn_f = np.ndarray(
        (len_valdn_f, n_imposters+1), 
        dtype=np.float32, buffer=pq_valdn_f_shm.buf)
    pq_valdn_f = torch.from_numpy(shared_pq_valdn_f)
    tp_valdn_f_shm = shared_memory.SharedMemory(name=tp_valdn_f_shm_name)
    shared_tp_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.int8, buffer=tp_valdn_f_shm.buf)
    tp_valdn_f = torch.from_numpy(shared_tp_valdn_f.astype(np.int32))
    log_w_valdn_f_shm = shared_memory.SharedMemory(name=log_w_valdn_f_shm_name)
    shared_log_w_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.float32, buffer=log_w_valdn_f_shm.buf)
    log_w_valdn_f = torch.from_numpy(shared_log_w_valdn_f)
    trend_valdn_f_shm = shared_memory.SharedMemory(name=trend_valdn_f_shm_name)
    shared_trend_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.float32, buffer=trend_valdn_f_shm.buf)
    trend_valdn_f = torch.from_numpy(shared_trend_valdn_f)
    delta_valdn_f_shm = shared_memory.SharedMemory(name=delta_valdn_f_shm_name)
    shared_delta_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.int16, buffer=delta_valdn_f_shm.buf)
    delta_valdn_f = torch.from_numpy(shared_delta_valdn_f.astype(np.int32))
    yr_valdn_f_shm = shared_memory.SharedMemory(name=yr_valdn_f_shm_name)
    yr_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.int8, buffer=yr_valdn_f_shm.buf)

    q_valdn_coo_m_shm = shared_memory.SharedMemory(name=q_valdn_coo_m_shm_name)
    shared_q_valdn_coo_m = np.ndarray(
        (3, len_valdn_val_m), 
        dtype=np.int16, buffer=q_valdn_coo_m_shm.buf)
    q_valdn_coo_m = torch.from_numpy(shared_q_valdn_coo_m.astype(np.int32))
    q_valdn_val_m_shm = shared_memory.SharedMemory(name=q_valdn_val_m_shm_name)
    shared_q_valdn_val_m = np.ndarray(
        (len_valdn_val_m, ), 
        dtype=np.int16, buffer=q_valdn_val_m_shm.buf)
    q_valdn_val_m = torch.from_numpy(shared_q_valdn_val_m.astype(np.float32))
    q_valdn_m = torch.sparse_coo_tensor(
        indices=q_valdn_coo_m,
        values=q_valdn_val_m,
        size=(len_valdn_m, n_upc_m, n_imposters+1)
    )
    pq_valdn_m_shm = shared_memory.SharedMemory(name=pq_valdn_m_shm_name)
    shared_pq_valdn_m = np.ndarray(
        (len_valdn_m, n_imposters+1), 
        dtype=np.float32, buffer=pq_valdn_m_shm.buf)
    pq_valdn_m = torch.from_numpy(shared_pq_valdn_m)
    tp_valdn_m_shm = shared_memory.SharedMemory(name=tp_valdn_m_shm_name)
    shared_tp_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.int8, buffer=tp_valdn_m_shm.buf)
    tp_valdn_m = torch.from_numpy(shared_tp_valdn_m.astype(np.int32))
    log_w_valdn_m_shm = shared_memory.SharedMemory(name=log_w_valdn_m_shm_name)
    shared_log_w_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.float32, buffer=log_w_valdn_m_shm.buf)
    log_w_valdn_m = torch.from_numpy(shared_log_w_valdn_m)
    trend_valdn_m_shm = shared_memory.SharedMemory(name=trend_valdn_m_shm_name)
    shared_trend_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.float32, buffer=trend_valdn_m_shm.buf)
    trend_valdn_m = torch.from_numpy(shared_trend_valdn_m)
    delta_valdn_m_shm = shared_memory.SharedMemory(name=delta_valdn_m_shm_name)
    shared_delta_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.int16, buffer=delta_valdn_m_shm.buf)
    delta_valdn_m = torch.from_numpy(shared_delta_valdn_m.astype(np.int32))
    yr_valdn_m_shm = shared_memory.SharedMemory(name=yr_valdn_m_shm_name)
    yr_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.int8, buffer=yr_valdn_m_shm.buf)

    q_valdn_coo_c_shm = shared_memory.SharedMemory(name=q_valdn_coo_c_shm_name)
    shared_q_valdn_coo_c = np.ndarray(
        (3, len_valdn_val_c), 
        dtype=np.int16, buffer=q_valdn_coo_c_shm.buf)
    q_valdn_coo_c = torch.from_numpy(shared_q_valdn_coo_c.astype(np.int32))
    q_valdn_val_c_shm = shared_memory.SharedMemory(name=q_valdn_val_c_shm_name)
    shared_q_valdn_val_c = np.ndarray(
        (len_valdn_val_c, ), 
        dtype=np.int16, buffer=q_valdn_val_c_shm.buf)
    q_valdn_val_c = torch.from_numpy(shared_q_valdn_val_c.astype(np.float32))
    q_valdn_c = torch.sparse_coo_tensor(
        indices=q_valdn_coo_c,
        values=q_valdn_val_c,
        size=(len_valdn_c, n_upc_c, n_imposters+1)
    )
    pq_valdn_c_shm = shared_memory.SharedMemory(name=pq_valdn_c_shm_name)
    shared_pq_valdn_c = np.ndarray(
        (len_valdn_c, n_imposters+1), 
        dtype=np.float32, buffer=pq_valdn_c_shm.buf)
    pq_valdn_c = torch.from_numpy(shared_pq_valdn_c)
    tp_valdn_c_shm = shared_memory.SharedMemory(name=tp_valdn_c_shm_name)
    shared_tp_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.int8, buffer=tp_valdn_c_shm.buf)
    tp_valdn_c = torch.from_numpy(shared_tp_valdn_c.astype(np.int32))
    log_w_valdn_c_shm = shared_memory.SharedMemory(name=log_w_valdn_c_shm_name)
    shared_log_w_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.float32, buffer=log_w_valdn_c_shm.buf)
    log_w_valdn_c = torch.from_numpy(shared_log_w_valdn_c)
    trend_valdn_c_shm = shared_memory.SharedMemory(name=trend_valdn_c_shm_name)
    shared_trend_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.float32, buffer=trend_valdn_c_shm.buf)
    trend_valdn_c = torch.from_numpy(shared_trend_valdn_c)
    delta_valdn_c_shm = shared_memory.SharedMemory(name=delta_valdn_c_shm_name)
    shared_delta_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.int16, buffer=delta_valdn_c_shm.buf)
    delta_valdn_c = torch.from_numpy(shared_delta_valdn_c.astype(np.int32))
    yr_valdn_c_shm = shared_memory.SharedMemory(name=yr_valdn_c_shm_name)
    yr_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.int8, buffer=yr_valdn_c_shm.buf)

    # Generate bootstrap samples
    idx_bt_f = idx_bt[0].to(torch.int64)
    idx_bt_m = idx_bt[1].to(torch.int64)
    idx_bt_c = idx_bt[2].to(torch.int64)

    q_bt_f = torch.sparse_coo_tensor(
        indices=torch.stack((row_f, col_f)),
        values=q_val_f,
        size=(nobs_f, n_upc_f)
    ).index_select(0, idx_bt_f)
    p_bt_f = torch.sparse_coo_tensor(
        indices=torch.stack((row_f, col_f)),
        values=p_val_f,
        size=(nobs_f, n_upc_f)
    ).index_select(0, idx_bt_f)
    row_bt_f = q_bt_f.coalesce().indices()[0, :]
    col_bt_f = q_bt_f.coalesce().indices()[1, :]
    q_bt_val_f = q_bt_f.coalesce().values()
    p_bt_val_f = p_bt_f.coalesce().values()
    yr_bt_idx_f = yr_idx_f[idx_bt_f]
    tp_bt_idx_f = tp_idx_f[idx_bt_f]
    delta_bt_idx_f = delta_idx_f[idx_bt_f]
    log_w_bt_f = log_w_f[idx_bt_f]
    trend_bt_f = trend_f[idx_bt_f]

    q_bt_m = torch.sparse_coo_tensor(
        indices=torch.stack((row_m, col_m)),
        values=q_val_m,
        size=(nobs_m, n_upc_m)
    ).index_select(0, idx_bt_m)
    p_bt_m = torch.sparse_coo_tensor(
        indices=torch.stack((row_m, col_m)),
        values=p_val_m,
        size=(nobs_m, n_upc_m)
    ).index_select(0, idx_bt_m)
    row_bt_m = q_bt_m.coalesce().indices()[0, :]
    col_bt_m = q_bt_m.coalesce().indices()[1, :]
    q_bt_val_m = q_bt_m.coalesce().values()
    p_bt_val_m = p_bt_m.coalesce().values()
    yr_bt_idx_m = yr_idx_m[idx_bt_m]
    tp_bt_idx_m = tp_idx_m[idx_bt_m]
    delta_bt_idx_m = delta_idx_m[idx_bt_m]
    log_w_bt_m = log_w_m[idx_bt_m]
    trend_bt_m = trend_m[idx_bt_m]

    q_bt_c = torch.sparse_coo_tensor(
        indices=torch.stack((row_c, col_c)),
        values=q_val_c,
        size=(nobs_c, n_upc_c)
    ).index_select(0, idx_bt_c)
    p_bt_c = torch.sparse_coo_tensor(
        indices=torch.stack((row_c, col_c)),
        values=p_val_c,
        size=(nobs_c, n_upc_c)
    ).index_select(0, idx_bt_c)
    row_bt_c = q_bt_c.coalesce().indices()[0, :]
    col_bt_c = q_bt_c.coalesce().indices()[1, :]
    q_bt_val_c = q_bt_c.coalesce().values()
    p_bt_val_c = p_bt_c.coalesce().values()
    yr_bt_idx_c = yr_idx_c[idx_bt_c]
    tp_bt_idx_c = tp_idx_c[idx_bt_c]
    delta_bt_idx_c = delta_idx_c[idx_bt_c]
    log_w_bt_c = log_w_c[idx_bt_c]
    trend_bt_c = trend_c[idx_bt_c]

    # Create imposters: female
    row_bt_im_f = row_bt_f.repeat(n_imposters)
    col_bt_im_f = col_bt_f.repeat(n_imposters)
    q_bt_im_val_f = q_bt_val_f.repeat(n_imposters)
    p_bt_im_val_f = p_bt_val_f.repeat(n_imposters)

    for i in range(len_train_f):
        row_idx = (row_bt_f==i).nonzero(as_tuple=True)[0]
        item_bought = col_bt_f[row_idx]
        candidate = candidate_f[yr_bt_idx_f[i]]
        candidate = candidate[~torch.isin(candidate, item_bought)]
        if len(row_idx) > 1:
            idx_del = row_idx[
                torch.randint(high=len(row_idx), size=(1, ), device=device)]
            idx_del = torch.tensor(
                [x*len(col_bt_f)+idx_del for x in range(n_imposters)], 
                device=device)
        else:
            idx_del = torch.tensor(
                [x*len(col_bt_f)+row_idx for x in range(n_imposters)], 
                device=device)
        selected_idx = torch.randperm(len(candidate), 
                                      device=device)[:n_imposters]
        selected = candidate[selected_idx]
        quantity = torch.randint(
            1, max_q+1, size=(n_imposters, ), device=device)
        price = price_f[yr_bt_idx_f[i], selected]
        col_bt_im_f[idx_del] = selected
        q_bt_im_val_f[idx_del] = quantity.to(dtype=torch.float32)
        p_bt_im_val_f[idx_del] = price

    row_bt_im_f = torch.cat((row_bt_f, row_bt_im_f))
    col_bt_im_f = torch.cat((col_bt_f, col_bt_im_f))
    q_bt_im_val_f = torch.cat((q_bt_val_f, q_bt_im_val_f))
    p_bt_im_val_f = torch.cat((p_bt_val_f, p_bt_im_val_f))
    pq_bt_im_val_f = torch.split(q_bt_im_val_f*p_bt_im_val_f, len(row_bt_f))
    bt_imp_idx_f = torch.arange(
        n_imposters+1, device=device).repeat_interleave(len(row_bt_f))

    pq_bt_im_f = torch.zeros((len_train_f, n_imposters+1), device=device)

    for i in range(n_imposters+1):
        pq_bt_im_f[:, i] = pq_bt_im_f[:, i].scatter_add(
            0, row_bt_f, pq_bt_im_val_f[i])

    # Create imposters: male
    row_bt_im_m = row_bt_m.repeat(n_imposters)
    col_bt_im_m = col_bt_m.repeat(n_imposters)
    q_bt_im_val_m = q_bt_val_m.repeat(n_imposters)
    p_bt_im_val_m = p_bt_val_m.repeat(n_imposters)

    for i in range(len_train_m):
        row_idx = (row_bt_m==i).nonzero(as_tuple=True)[0]
        item_bought = col_bt_m[row_idx]
        candidate = candidate_m[yr_bt_idx_m[i]]
        candidate = candidate[~torch.isin(candidate, item_bought)]
        if len(row_idx) > 1:
            idx_del = row_idx[
                torch.randint(high=len(row_idx), size=(1, ), device=device)]
            idx_del = torch.tensor(
                [x*len(col_bt_m)+idx_del for x in range(n_imposters)], 
                device=device)
        else:
            idx_del = torch.tensor(
                [x*len(col_bt_m)+row_idx for x in range(n_imposters)], 
                device=device)
        selected_idx = torch.randperm(len(candidate), 
                                      device=device)[:n_imposters]
        selected = candidate[selected_idx]
        quantity = torch.randint(
            1, max_q+1, size=(n_imposters, ), device=device)
        price = price_m[yr_bt_idx_m[i], selected]
        col_bt_im_m[idx_del] = selected
        q_bt_im_val_m[idx_del] = quantity.to(dtype=torch.float32)
        p_bt_im_val_m[idx_del] = price

    row_bt_im_m = torch.cat((row_bt_m, row_bt_im_m))
    col_bt_im_m = torch.cat((col_bt_m, col_bt_im_m))
    q_bt_im_val_m = torch.cat((q_bt_val_m, q_bt_im_val_m))
    p_bt_im_val_m = torch.cat((p_bt_val_m, p_bt_im_val_m))
    pq_bt_im_val_m = torch.split(q_bt_im_val_m*p_bt_im_val_m, len(row_bt_m))
    bt_imp_idx_m = torch.arange(
        n_imposters+1, device=device).repeat_interleave(len(row_bt_m))

    pq_bt_im_m = torch.zeros((len_train_m, n_imposters+1), device=device)

    for i in range(n_imposters+1):
        pq_bt_im_m[:, i] = pq_bt_im_m[:, i].scatter_add(
            0, row_bt_m, pq_bt_im_val_m[i])

    # Create imposters: children
    row_bt_im_c = row_bt_c.repeat(n_imposters)
    col_bt_im_c = col_bt_c.repeat(n_imposters)
    q_bt_im_val_c = q_bt_val_c.repeat(n_imposters)
    p_bt_im_val_c = p_bt_val_c.repeat(n_imposters)

    for i in range(len_train_c):
        row_idx = (row_bt_c==i).nonzero(as_tuple=True)[0]
        item_bought = col_bt_c[row_idx]
        candidate = candidate_c[yr_bt_idx_c[i]]
        candidate = candidate[~torch.isin(candidate, item_bought)]
        if len(row_idx) > 1:
            idx_del = row_idx[
                torch.randint(high=len(row_idx), size=(1, ), device=device)]
            idx_del = torch.tensor(
                [x*len(col_bt_c)+idx_del for x in range(n_imposters)], 
                device=device)
        else:
            idx_del = torch.tensor(
                [x*len(col_bt_c)+row_idx for x in range(n_imposters)], 
                device=device)
        selected_idx = torch.randperm(len(candidate), 
                                      device=device)[:n_imposters]
        selected = candidate[selected_idx]
        quantity = torch.randint(
            1, max_q+1, size=(n_imposters, ), device=device)
        price = price_c[yr_bt_idx_c[i], selected]
        col_bt_im_c[idx_del] = selected
        q_bt_im_val_c[idx_del] = quantity.to(dtype=torch.float32)
        p_bt_im_val_c[idx_del] = price

    row_bt_im_c = torch.cat((row_bt_c, row_bt_im_c))
    col_bt_im_c = torch.cat((col_bt_c, col_bt_im_c))
    q_bt_im_val_c = torch.cat((q_bt_val_c, q_bt_im_val_c))
    p_bt_im_val_c = torch.cat((p_bt_val_c, p_bt_im_val_c))
    pq_bt_im_val_c = torch.split(q_bt_im_val_c*p_bt_im_val_c, len(row_bt_c))
    bt_imp_idx_c = torch.arange(
        n_imposters+1, device=device).repeat_interleave(len(row_bt_c))

    pq_bt_im_c = torch.zeros((len_train_c, n_imposters+1), device=device)

    for i in range(n_imposters+1):
        pq_bt_im_c[:, i] = pq_bt_im_c[:, i].scatter_add(
            0, row_bt_c, pq_bt_im_val_c[i])

    # Model parameters
    params_bt = init.clone().requires_grad_()

    # Gradient descent
    m = torch.zeros_like(init, device=device)
    v = torch.zeros_like(init, device=device)
    t = 0
    batch = 0
    epoch = 1
    best_valdn_loss = torch.tensor([float('inf')], device=device)
    no_improvement_counter = 0
    improvement_t = torch.zeros(max_epoch, device=device)

    while epoch <= max_epoch:
        idx_selected_f = bt_idx_f[batch]
        f_batch = torch.isin(row_bt_im_f, idx_selected_f)
        q_batch_f = torch.sparse_coo_tensor(
            indices=torch.stack((
                torch.searchsorted(idx_selected_f, row_bt_im_f[f_batch]), 
                col_bt_im_f[f_batch], 
                bt_imp_idx_f[f_batch])),
            values=q_bt_im_val_f[f_batch],
            size=(batch_size_f, n_upc_f, n_imposters+1)
        )
        pq_batch_f = pq_bt_im_f[idx_selected_f]
        tp_batch_f = tp_bt_idx_f[idx_selected_f]
        delta_batch_f = delta_bt_idx_f[idx_selected_f]
        log_w_batch_f = log_w_bt_f[idx_selected_f]
        trend_batch_f = trend_bt_f[idx_selected_f]
        yr_batch_f = yr_bt_idx_f[idx_selected_f]

        idx_selected_m = bt_idx_m[batch]
        m_batch = torch.isin(row_bt_im_m, idx_selected_m)
        q_batch_m = torch.sparse_coo_tensor(
            indices=torch.stack((
                torch.searchsorted(idx_selected_m, row_bt_im_m[m_batch]), 
                col_bt_im_m[m_batch], 
                bt_imp_idx_m[m_batch])),
            values=q_bt_im_val_m[m_batch],
            size=(batch_size_m, n_upc_m, n_imposters+1)
        )
        pq_batch_m = pq_bt_im_m[idx_selected_m]
        tp_batch_m = tp_bt_idx_m[idx_selected_m]
        delta_batch_m = delta_bt_idx_m[idx_selected_m]
        log_w_batch_m = log_w_bt_m[idx_selected_m]
        trend_batch_m = trend_bt_m[idx_selected_m]
        yr_batch_m = yr_bt_idx_m[idx_selected_m]

        idx_selected_c = bt_idx_c[batch]
        c_batch = torch.isin(row_bt_im_c, idx_selected_c)
        q_batch_c = torch.sparse_coo_tensor(
            indices=torch.stack((
                torch.searchsorted(idx_selected_c, row_bt_im_c[c_batch]), 
                col_bt_im_c[c_batch], 
                bt_imp_idx_c[c_batch])),
            values=q_bt_im_val_c[c_batch],
            size=(batch_size_c, n_upc_c, n_imposters+1)
        )
        pq_batch_c = pq_bt_im_c[idx_selected_c]
        tp_batch_c = tp_bt_idx_c[idx_selected_c]
        delta_batch_c = delta_bt_idx_c[idx_selected_c]
        log_w_batch_c = log_w_bt_c[idx_selected_c]
        trend_batch_c = trend_bt_c[idx_selected_c]
        yr_batch_c = yr_bt_idx_c[idx_selected_c]

        loss = mle(params_bt, q_batch_f, q_batch_m, q_batch_c, 
                   pq_batch_f, pq_batch_m, pq_batch_c, 
                   tp_batch_f, tp_batch_m, tp_batch_c, 
                   log_w_batch_f, log_w_batch_m, log_w_batch_c,
                   delta_batch_f, delta_batch_m, delta_batch_c,
                   trend_batch_f, trend_batch_m, trend_batch_c,
                   yr_batch_f, yr_batch_m, yr_batch_c,
                   param_idx, n_attr_f, n_attr_m, n_attr_c,
                   n_upc_f, n_upc_m, n_upc_c, delta_idx_2004, device)
        loss.backward()

        with torch.no_grad():
            gradient = params_bt.grad
            t += 1
            m = beta_1*m + (1-beta_1)*gradient
            v = beta_2*v + (1-beta_2)*(gradient**2)
            m_hat = m / (1-beta_1**t)
            v_hat = v / (1-beta_2**t)
            params_bt -= lr*m_hat / (torch.sqrt(v_hat)+epsilon)
            if (epoch == max_epoch) & ((batch+1) == n_batch):
                break
    
        params_bt.grad.zero_()
        batch = (batch+1) % n_batch
        if batch == 0:
            # Validation
            with torch.no_grad():
                valdn_loss = mle(params_bt, q_valdn_f, q_valdn_m, q_valdn_c, 
                                 pq_valdn_f, pq_valdn_m, pq_valdn_c, 
                                 tp_valdn_f, tp_valdn_m, tp_valdn_c, 
                                 log_w_valdn_f, log_w_valdn_m, log_w_valdn_c,
                                 delta_valdn_f, delta_valdn_m, delta_valdn_c, 
                                 trend_valdn_f, trend_valdn_m, trend_valdn_c,
                                 yr_valdn_f, yr_valdn_m, yr_valdn_c,
                                 param_idx, n_attr_f, n_attr_m, 
                                 n_attr_c, n_upc_f, n_upc_m, n_upc_c, 
                                 delta_idx_2004, device).item()
                if valdn_loss < best_valdn_loss:
                    best_valdn_loss = valdn_loss
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
                improvement_t[epoch-1] = no_improvement_counter
                if no_improvement_counter == patience:
                    break

            epoch += 1

            # Generate new imposters: female
            col_bt_im_f = col_bt_f.repeat(n_imposters)
            q_bt_im_val_f = q_bt_val_f.repeat(n_imposters)
            p_bt_im_val_f = p_bt_val_f.repeat(n_imposters)

            for i in range(len_train_f):
                row_idx = (row_bt_f==i).nonzero(as_tuple=True)[0]
                item_bought = col_bt_f[row_idx]
                candidate = candidate_f[yr_bt_idx_f[i]]
                candidate = candidate[~torch.isin(candidate, item_bought)]
                if len(row_idx) > 1:
                    idx_del = row_idx[torch.randint(
                        high=len(row_idx), size=(1, ), device=device)]
                    idx_del = torch.tensor(
                        [x*len(col_bt_f)+idx_del for x in range(n_imposters)], 
                        device=device)
                else:
                    idx_del = torch.tensor(
                        [x*len(col_bt_f)+row_idx for x in range(n_imposters)], 
                        device=device)
                selected_idx = torch.randperm(
                    len(candidate), device=device)[:n_imposters]
                selected = candidate[selected_idx]
                quantity = torch.randint(
                    1, max_q+1, size=(n_imposters, ), device=device)
                price = price_f[yr_bt_idx_f[i], selected]
                col_bt_im_f[idx_del] = selected
                q_bt_im_val_f[idx_del] = quantity.to(dtype=torch.float32)
                p_bt_im_val_f[idx_del] = price

            col_bt_im_f = torch.cat((col_bt_f, col_bt_im_f))
            q_bt_im_val_f = torch.cat((q_bt_val_f, q_bt_im_val_f))
            p_bt_im_val_f = torch.cat((p_bt_val_f, p_bt_im_val_f))
            pq_bt_im_val_f = torch.split(
                q_bt_im_val_f*p_bt_im_val_f, len(row_bt_f))
            
            pq_bt_im_f = torch.zeros(
                (len_train_f, n_imposters+1), device=device)

            for i in range(n_imposters+1):
                pq_bt_im_f[:, i] = pq_bt_im_f[:, i].scatter_add(
                    0, row_bt_f, pq_bt_im_val_f[i])

            # Generate new imposters: male
            col_bt_im_m = col_bt_m.repeat(n_imposters)
            q_bt_im_val_m = q_bt_val_m.repeat(n_imposters)
            p_bt_im_val_m = p_bt_val_m.repeat(n_imposters)

            for i in range(len_train_m):
                row_idx = (row_bt_m==i).nonzero(as_tuple=True)[0]
                item_bought = col_bt_m[row_idx]
                candidate = candidate_m[yr_bt_idx_m[i]]
                candidate = candidate[~torch.isin(candidate, item_bought)]
                if len(row_idx) > 1:
                    idx_del = row_idx[torch.randint(
                        high=len(row_idx), size=(1, ), device=device)]
                    idx_del = torch.tensor(
                        [x*len(col_bt_m)+idx_del for x in range(n_imposters)], 
                        device=device)
                else:
                    idx_del = torch.tensor(
                        [x*len(col_bt_m)+row_idx for x in range(n_imposters)], 
                        device=device)
                selected_idx = torch.randperm(
                    len(candidate), device=device)[:n_imposters]
                selected = candidate[selected_idx]
                quantity = torch.randint(
                    1, max_q+1, size=(n_imposters, ), device=device)
                price = price_m[yr_bt_idx_m[i], selected]
                col_bt_im_m[idx_del] = selected
                q_bt_im_val_m[idx_del] = quantity.to(dtype=torch.float32)
                p_bt_im_val_m[idx_del] = price

            col_bt_im_m = torch.cat((col_bt_m, col_bt_im_m))
            q_bt_im_val_m = torch.cat((q_bt_val_m, q_bt_im_val_m))
            p_bt_im_val_m = torch.cat((p_bt_val_m, p_bt_im_val_m))
            pq_bt_im_val_m = torch.split(
                q_bt_im_val_m*p_bt_im_val_m, len(row_bt_m))
            
            pq_bt_im_m = torch.zeros(
                (len_train_m, n_imposters+1), device=device)

            for i in range(n_imposters+1):
                pq_bt_im_m[:, i] = pq_bt_im_m[:, i].scatter_add(
                    0, row_bt_m, pq_bt_im_val_m[i])

            # Generate new imposters: children
            col_bt_im_c = col_bt_c.repeat(n_imposters)
            q_bt_im_val_c = q_bt_val_c.repeat(n_imposters)
            p_bt_im_val_c = p_bt_val_c.repeat(n_imposters)

            for i in range(len_train_c):
                row_idx = (row_bt_c==i).nonzero(as_tuple=True)[0]
                item_bought = col_bt_c[row_idx]
                candidate = candidate_c[yr_bt_idx_c[i]]
                candidate = candidate[~torch.isin(candidate, item_bought)]
                if len(row_idx) > 1:
                    idx_del = row_idx[torch.randint(
                        high=len(row_idx), size=(1, ), device=device)]
                    idx_del = torch.tensor(
                        [x*len(col_bt_c)+idx_del for x in range(n_imposters)], 
                        device=device)
                else:
                    idx_del = torch.tensor(
                        [x*len(col_bt_c)+row_idx for x in range(n_imposters)], 
                        device=device)
                selected_idx = torch.randperm(
                    len(candidate), device=device)[:n_imposters]
                selected = candidate[selected_idx]
                quantity = torch.randint(
                    1, max_q+1, size=(n_imposters, ), device=device)
                price = price_c[yr_bt_idx_c[i], selected]
                col_bt_im_c[idx_del] = selected
                q_bt_im_val_c[idx_del] = quantity.to(dtype=torch.float32)
                p_bt_im_val_c[idx_del] = price

            col_bt_im_c = torch.cat((col_bt_c, col_bt_im_c))
            q_bt_im_val_c = torch.cat((q_bt_val_c, q_bt_im_val_c))
            p_bt_im_val_c = torch.cat((p_bt_val_c, p_bt_im_val_c))
            pq_bt_im_val_c = torch.split(
                q_bt_im_val_c*p_bt_im_val_c, len(row_bt_c))
            
            pq_bt_im_c = torch.zeros(
                (len_train_c, n_imposters+1), device=device)

            for i in range(n_imposters+1):
                pq_bt_im_c[:, i] = pq_bt_im_c[:, i].scatter_add(
                    0, row_bt_c, pq_bt_im_val_c[i])

    return params_bt[
        :, range(param_idx[11], param_idx[16])].detach().squeeze()

def resource_share(delta_est, lambda_w_est, lambda_yr_est, lambda_yr_w_est, 
                   beta2_cp_est, beta2_one_est, beta2_two_est):
    delta_dict = {}
    j = 0

    for key in delta_idx.keys():
        gender, year, type_ = key.split("_")
        if (gender == '0') & (type_ == '0'):
            delta_est_extr = delta_est[0].item()
            lambda_w_est_extr = lambda_w_est[0].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[0].item()
        elif (gender == '0') & (type_ == '1'):
            delta_est_extr = delta_est[1].item()
            lambda_w_est_extr = lambda_w_est[1].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[1].item()
        elif (gender == '0') & (type_ == '2'):
            delta_est_extr = delta_est[2].item()
            lambda_w_est_extr = lambda_w_est[2].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[2].item()
        elif (gender == '1') & (type_ == '0'):
            delta_est_extr = delta_est[3].item()
            lambda_w_est_extr = lambda_w_est[3].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[3].item()
        elif (gender == '1') & (type_ == '1'):
            delta_est_extr = delta_est[4].item()
            lambda_w_est_extr = lambda_w_est[4].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[4].item()
        elif (gender == '1') & (type_ == '2'):
            delta_est_extr = delta_est[5].item()
            lambda_w_est_extr = lambda_w_est[5].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[5].item()
        elif (gender == '2') & (type_ == '1'):
            delta_est_extr = delta_est[6].item()
            lambda_w_est_extr = lambda_w_est[6].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[6].item()
        else:
            delta_est_extr = delta_est[7].item()
            lambda_w_est_extr = lambda_w_est[7].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[7].item()
        if year != '2004':
            lambda_yr_est_extr = lambda_yr_est[j].item()
            delta_dict[key] = {
                'delta_est': delta_est_extr,
                'lambda_w_est': lambda_w_est_extr,
                'lambda_yr_est': lambda_yr_est_extr,
                'lambda_yr_w_est': lambda_yr_w_est_extr
            }
            j += 1
        else:
            delta_dict[key] = {
                'delta_est': delta_est_extr,
                'lambda_w_est': lambda_w_est_extr,
            }

    w = np.array([1875, 2292, 2708, 3125, 3542, 3958, 4583, 5417])

    w_demean = np.array([np.log(1875) - log_w_mean,
                         np.log(2292) - log_w_mean,
                         np.log(2708) - log_w_mean,
                         np.log(3125) - log_w_mean,
                         np.log(3542) - log_w_mean,
                         np.log(3958) - log_w_mean,
                         np.log(4583) - log_w_mean,
                         np.log(5417) - log_w_mean])
    
    trend = {}

    for i, yr in enumerate(range(2004, 2021)):
        trend[str(yr)] = i / (len(range(2004, 2021))-1)

    yr_tp = defaultdict(list)

    for key, value in delta_dict.items():
        gender, year, type_ = key.split("_")
        group_key = f"{year}_{type_}"
        yr_tp[group_key].append(value)

    shares = {}

    for key, values in yr_tp.items():
        year, type_ = key.split("_")
        if type_ == '0':
            delta_est_f = np.float64(values[0]['delta_est'])
            delta_est_m = np.float64(values[1]['delta_est'])
            lambda_w_est_f = np.float64(values[0]['lambda_w_est'])
            lambda_w_est_m = np.float64(values[1]['lambda_w_est'])
            if year == '2004':
                delta_est_f = np.exp(delta_est_f + lambda_w_est_f*w_demean)
                delta_est_m = np.exp(delta_est_m + lambda_w_est_m*w_demean)
            else:
                lambda_yr_est_f = np.float64(values[0]['lambda_yr_est'])
                lambda_yr_est_m = np.float64(values[1]['lambda_yr_est'])
                lambda_yr_w_est_f = np.float64(values[0]['lambda_yr_w_est'])
                lambda_yr_w_est_m = np.float64(values[1]['lambda_yr_w_est'])
                delta_est_f = np.exp(delta_est_f + lambda_yr_est_f 
                                    + lambda_w_est_f*w_demean 
                                    + lambda_yr_w_est_f*w_demean*trend[year])
                delta_est_m = np.exp(delta_est_m + lambda_yr_est_m 
                                    + lambda_w_est_m*w_demean 
                                    + lambda_yr_w_est_m*w_demean*trend[year])
            beta1_est = (delta_est_f+delta_est_m+2*beta2_cp_est*w) / 2
            share_f = (beta1_est-delta_est_f) / (2*beta2_cp_est*w)
            share_m = (beta1_est-delta_est_m) / (2*beta2_cp_est*w)
            shares[key] = {'female': share_f, 'male': share_m}
        elif type_ == '1':
            delta_est_f = np.float64(values[0]['delta_est'])
            delta_est_m = np.float64(values[1]['delta_est'])
            delta_est_c = np.float64(values[2]['delta_est'])
            lambda_w_est_f = np.float64(values[0]['lambda_w_est'])
            lambda_w_est_m = np.float64(values[1]['lambda_w_est'])
            lambda_w_est_c = np.float64(values[2]['lambda_w_est'])
            if year == '2004':
                delta_est_f = np.exp(delta_est_f + lambda_w_est_f*w_demean)
                delta_est_m = np.exp(delta_est_m + lambda_w_est_m*w_demean)
                delta_est_c = np.exp(delta_est_c + lambda_w_est_c*w_demean)
            else:
                lambda_yr_est_f = np.float64(values[0]['lambda_yr_est'])
                lambda_yr_est_m = np.float64(values[1]['lambda_yr_est'])
                lambda_yr_est_c = np.float64(values[2]['lambda_yr_est'])
                lambda_yr_w_est_f = np.float64(values[0]['lambda_yr_w_est'])
                lambda_yr_w_est_m = np.float64(values[1]['lambda_yr_w_est'])
                lambda_yr_w_est_c = np.float64(values[2]['lambda_yr_w_est'])
                delta_est_f = np.exp(delta_est_f + lambda_yr_est_f 
                                    + lambda_w_est_f*w_demean 
                                    + lambda_yr_w_est_f*w_demean*trend[year])
                delta_est_m = np.exp(delta_est_m + lambda_yr_est_m 
                                    + lambda_w_est_m*w_demean 
                                    + lambda_yr_w_est_m*w_demean*trend[year])
                delta_est_c = np.exp(delta_est_c + lambda_yr_est_c 
                                    + lambda_w_est_c*w_demean 
                                    + lambda_yr_w_est_c*w_demean*trend[year])
            beta1_est = (delta_est_f+
                         delta_est_m+
                         delta_est_c+
                         2*beta2_one_est*w) / 3
            share_f = (beta1_est-delta_est_f) / (2*beta2_one_est*w)
            share_m = (beta1_est-delta_est_m) / (2*beta2_one_est*w)
            share_c = (beta1_est-delta_est_c) / (2*beta2_one_est*w)
            shares[key] = {'female': share_f, 
                           'male': share_m, 
                           'children': share_c}
        else:
            delta_est_f = np.float64(values[0]['delta_est'])
            delta_est_m = np.float64(values[1]['delta_est'])
            delta_est_c = np.float64(values[2]['delta_est'])
            lambda_w_est_f = np.float64(values[0]['lambda_w_est'])
            lambda_w_est_m = np.float64(values[1]['lambda_w_est'])
            lambda_w_est_c = np.float64(values[2]['lambda_w_est'])
            if year == '2004':
                delta_est_f = np.exp(delta_est_f + lambda_w_est_f*w_demean)
                delta_est_m = np.exp(delta_est_m + lambda_w_est_m*w_demean)
                delta_est_c = np.exp(delta_est_c + lambda_w_est_c*w_demean)
            else:
                lambda_yr_est_f = np.float64(values[0]['lambda_yr_est'])
                lambda_yr_est_m = np.float64(values[1]['lambda_yr_est'])
                lambda_yr_est_c = np.float64(values[2]['lambda_yr_est'])
                lambda_yr_w_est_f = np.float64(values[0]['lambda_yr_w_est'])
                lambda_yr_w_est_m = np.float64(values[1]['lambda_yr_w_est'])
                lambda_yr_w_est_c = np.float64(values[2]['lambda_yr_w_est'])
                delta_est_f = np.exp(delta_est_f + lambda_yr_est_f 
                                    + lambda_w_est_f*w_demean 
                                    + lambda_yr_w_est_f*w_demean*trend[year])
                delta_est_m = np.exp(delta_est_m + lambda_yr_est_m 
                                    + lambda_w_est_m*w_demean 
                                    + lambda_yr_w_est_m*w_demean*trend[year])
                delta_est_c = np.exp(delta_est_c + lambda_yr_est_c 
                                    + lambda_w_est_c*w_demean 
                                    + lambda_yr_w_est_c*w_demean*trend[year])
            beta1_est = (delta_est_f+
                         delta_est_m+
                         delta_est_c+
                         2*beta2_two_est*w) / 3
            share_f = (beta1_est-delta_est_f) / (2*beta2_two_est*w)
            share_m = (beta1_est-delta_est_m) / (2*beta2_two_est*w)
            share_c = (beta1_est-delta_est_c) / (2*beta2_two_est*w)
            shares[key] = {'female': share_f, 
                           'male': share_m, 
                           'children': share_c}
        
    # Average resource shares for each type_income group
    shares_tp_w = {}
    for key, values in shares.items():
        year, type_ = key.split("_")
        for income in range(len(w)):
            tp_w = f"{type_}_{income}"
            if type_ == '0':
                if tp_w not in shares_tp_w:
                    shares_tp_w[tp_w] = {'female': [], 'male': []}
                for k in values:
                    shares_tp_w[tp_w][k].append(values[k][income])
            else:
                if tp_w not in shares_tp_w:
                    shares_tp_w[tp_w] = {'female': [], 
                                         'male': [], 
                                         'children': []}
                for k in values:
                    shares_tp_w[tp_w][k].append(values[k][income])

    unreasonable_shares = {}

    for key, values in shares_tp_w.items():
        type_, income = key.split("_")
        neg_shares_f = [i for i, x in enumerate(values['female']) if x <= 0]
        neg_shares_m = [i for i, x in enumerate(values['male']) if x <= 0]
        if type_ != '0':
            neg_shares_c = [
                i for i, x in enumerate(values['children']) if x <= 0]
            neg_shares = list(set(neg_shares_f + neg_shares_m + neg_shares_c))
            values['female'] = [
                x for i, x in enumerate(values['female']) 
                if i not in neg_shares]
            values['male'] = [
                x for i, x in enumerate(values['male']) if i not in neg_shares]
            values['children'] = [
                x for i, x in enumerate(values['children']) 
                if i not in neg_shares]
        else:
            neg_shares = list(set(neg_shares_f + neg_shares_m))
            values['female'] = [
                x for i, x in enumerate(values['female']) 
                if i not in neg_shares]
            values['male'] = [
                x for i, x in enumerate(values['male']) if i not in neg_shares]
        if len(neg_shares) > 0:
            unreasonable_shares[key] = neg_shares

    avg_shares_tp_w = {
        tp_w: {k: sum(v) / len(v) for k, v in vals.items()}
        for tp_w, vals in shares_tp_w.items()
    }

    # Average resource shares for each type
    shares_tp = {}
    for key, values in shares.items():
        _, type_ = key.split("_")
        for income in range(len(w)):
            if type_ == '0':
                if type_ not in shares_tp:
                    shares_tp[type_] = {'female': [], 'male': []}
                for k in values:
                    shares_tp[type_][k].append(values[k][income])
            else:
                if type_ not in shares_tp:
                    shares_tp[type_] = {'female': [], 
                                        'male': [], 
                                        'children': []}
                for k in values:
                    shares_tp[type_][k].append(values[k][income])

    for key, values in shares_tp.items():
        neg_shares_f = [i for i, x in enumerate(values['female']) if x <= 0]
        neg_shares_m = [i for i, x in enumerate(values['male']) if x <= 0]
        if key != '0':
            neg_shares_c = [
                i for i, x in enumerate(values['children']) if x <= 0]
            neg_shares = list(set(neg_shares_f + neg_shares_m + neg_shares_c))
            values['female'] = [
                x for i, x in enumerate(values['female']) 
                if i not in neg_shares]
            values['male'] = [
                x for i, x in enumerate(values['male']) if i not in neg_shares]
            values['children'] = [
                x for i, x in enumerate(values['children']) 
                if i not in neg_shares]
        else:
            neg_shares = list(set(neg_shares_f + neg_shares_m))
            values['female'] = [
                x for i, x in enumerate(values['female']) 
                if i not in neg_shares]
            values['male'] = [
                x for i, x in enumerate(values['male']) if i not in neg_shares]

    avg_shares_tp = {
        tp: {k: sum(v) / len(v) for k, v in vals.items()}
        for tp, vals in shares_tp.items()
    }

    return shares, avg_shares_tp, avg_shares_tp_w, unreasonable_shares

def marginal_w(delta_est, lambda_w_est, lambda_yr_est, lambda_yr_w_est, 
               beta2_cp_est, beta2_one_est, beta2_two_est, w, w_demean):
    delta_dict = {}
    j = 0

    for key in delta_idx.keys():
        gender, year, type_ = key.split("_")
        if (gender == '0') & (type_ == '0'):
            delta_est_extr = delta_est[0].item()
            lambda_w_est_extr = lambda_w_est[0].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[0].item()
        elif (gender == '0') & (type_ == '1'):
            delta_est_extr = delta_est[1].item()
            lambda_w_est_extr = lambda_w_est[1].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[1].item()
        elif (gender == '0') & (type_ == '2'):
            delta_est_extr = delta_est[2].item()
            lambda_w_est_extr = lambda_w_est[2].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[2].item()
        elif (gender == '1') & (type_ == '0'):
            delta_est_extr = delta_est[3].item()
            lambda_w_est_extr = lambda_w_est[3].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[3].item()
        elif (gender == '1') & (type_ == '1'):
            delta_est_extr = delta_est[4].item()
            lambda_w_est_extr = lambda_w_est[4].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[4].item()
        elif (gender == '1') & (type_ == '2'):
            delta_est_extr = delta_est[5].item()
            lambda_w_est_extr = lambda_w_est[5].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[5].item()
        elif (gender == '2') & (type_ == '1'):
            delta_est_extr = delta_est[6].item()
            lambda_w_est_extr = lambda_w_est[6].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[6].item()
        else:
            delta_est_extr = delta_est[7].item()
            lambda_w_est_extr = lambda_w_est[7].item()
            lambda_yr_w_est_extr = lambda_yr_w_est[7].item()
        if year != '2004':
            lambda_yr_est_extr = lambda_yr_est[j].item()
            delta_dict[key] = {
                'delta_est': delta_est_extr,
                'lambda_w_est': lambda_w_est_extr,
                'lambda_yr_est': lambda_yr_est_extr,
                'lambda_yr_w_est': lambda_yr_w_est_extr
            }
            j += 1
        else:
            delta_dict[key] = {
                'delta_est': delta_est_extr,
                'lambda_w_est': lambda_w_est_extr,
            }
    
    trend = {}

    for i, yr in enumerate(range(2004, 2021)):
        trend[str(yr)] = i / (len(range(2004, 2021))-1)

    yr_tp = defaultdict(list)

    for key, value in delta_dict.items():
        gender, year, type_ = key.split("_")
        group_key = f"{year}_{type_}"
        yr_tp[group_key].append(value)

    marginal = {}

    for key, values in yr_tp.items():
        year, type_ = key.split("_")
        if type_ == '0':
            delta_est_f = np.float64(values[0]['delta_est'])
            delta_est_m = np.float64(values[1]['delta_est'])
            lambda_w_est_f = np.float64(values[0]['lambda_w_est'])
            lambda_w_est_m = np.float64(values[1]['lambda_w_est'])
            if year == '2004':
                delta_est_f = np.exp(delta_est_f + lambda_w_est_f*w_demean)
                delta_est_m = np.exp(delta_est_m + lambda_w_est_m*w_demean)
                part1 = (
                    (1/w)*(
                        lambda_w_est_f*delta_est_f+lambda_w_est_m*delta_est_m
                    )
                ) + 2*beta2_cp_est
                part2_f = (lambda_w_est_f-1)*delta_est_f + (
                    (delta_est_f+delta_est_m+2*beta2_cp_est*w)/2
                )
                part2_m = (lambda_w_est_m-1)*delta_est_m + (
                    (delta_est_f+delta_est_m+2*beta2_cp_est*w)/2
                )
            else:
                lambda_yr_est_f = np.float64(values[0]['lambda_yr_est'])
                lambda_yr_est_m = np.float64(values[1]['lambda_yr_est'])
                lambda_yr_w_est_f = np.float64(values[0]['lambda_yr_w_est'])
                lambda_yr_w_est_m = np.float64(values[1]['lambda_yr_w_est'])
                delta_est_f = np.exp(delta_est_f + lambda_yr_est_f 
                                    + lambda_w_est_f*w_demean 
                                    + lambda_yr_w_est_f*w_demean*trend[year])
                delta_est_m = np.exp(delta_est_m + lambda_yr_est_m 
                                    + lambda_w_est_m*w_demean 
                                    + lambda_yr_w_est_m*w_demean*trend[year])
                part1 = (
                    (1/w)*(
                        (lambda_w_est_f+lambda_yr_w_est_f*trend[year])
                        *delta_est_f
                        +(lambda_w_est_m+lambda_yr_w_est_m*trend[year])
                        *delta_est_m
                    )
                ) + 2*beta2_cp_est
                part2_f = (
                    lambda_w_est_f+lambda_yr_w_est_f*trend[year]-1
                )*delta_est_f + (
                    (delta_est_f+delta_est_m+2*beta2_cp_est*w)/2
                )
                part2_m = (
                    lambda_w_est_m+lambda_yr_w_est_m*trend[year]-1
                )*delta_est_m + (
                    (delta_est_f+delta_est_m+2*beta2_cp_est*w)/2
                )
            part1_wgh = 1/(2*2*beta2_cp_est*w)
            part2_wgh = 1/(2*beta2_cp_est*w*w)
            marginal_f = part1_wgh*part1 - part2_wgh*part2_f
            marginal_m = part1_wgh*part1 - part2_wgh*part2_m
            marginal[key] = {'female': marginal_f, 'male': marginal_m}
        elif type_ == '1':
            delta_est_f = np.float64(values[0]['delta_est'])
            delta_est_m = np.float64(values[1]['delta_est'])
            delta_est_c = np.float64(values[2]['delta_est'])
            lambda_w_est_f = np.float64(values[0]['lambda_w_est'])
            lambda_w_est_m = np.float64(values[1]['lambda_w_est'])
            lambda_w_est_c = np.float64(values[2]['lambda_w_est'])
            if year == '2004':
                delta_est_f = np.exp(delta_est_f + lambda_w_est_f*w_demean)
                delta_est_m = np.exp(delta_est_m + lambda_w_est_m*w_demean)
                delta_est_c = np.exp(delta_est_c + lambda_w_est_c*w_demean)
                part1 = (
                    (1/w)*(
                        lambda_w_est_f*delta_est_f
                        +lambda_w_est_m*delta_est_m
                        +lambda_w_est_c*delta_est_c
                    )
                ) + 2*beta2_one_est
                part2_f = (lambda_w_est_f-1)*delta_est_f + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_one_est*w)/3
                )
                part2_m = (lambda_w_est_m-1)*delta_est_m + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_one_est*w)/3
                )
                part2_c = (lambda_w_est_c-1)*delta_est_c + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_one_est*w)/3
                )
            else:
                lambda_yr_est_f = np.float64(values[0]['lambda_yr_est'])
                lambda_yr_est_m = np.float64(values[1]['lambda_yr_est'])
                lambda_yr_est_c = np.float64(values[2]['lambda_yr_est'])
                lambda_yr_w_est_f = np.float64(values[0]['lambda_yr_w_est'])
                lambda_yr_w_est_m = np.float64(values[1]['lambda_yr_w_est'])
                lambda_yr_w_est_c = np.float64(values[2]['lambda_yr_w_est'])
                delta_est_f = np.exp(delta_est_f + lambda_yr_est_f 
                                    + lambda_w_est_f*w_demean 
                                    + lambda_yr_w_est_f*w_demean*trend[year])
                delta_est_m = np.exp(delta_est_m + lambda_yr_est_m 
                                    + lambda_w_est_m*w_demean 
                                    + lambda_yr_w_est_m*w_demean*trend[year])
                delta_est_c = np.exp(delta_est_c + lambda_yr_est_c 
                                    + lambda_w_est_c*w_demean 
                                    + lambda_yr_w_est_c*w_demean*trend[year])
                part1 = (
                    (1/w)*(
                        (lambda_w_est_f+lambda_yr_w_est_f*trend[year])
                        *delta_est_f
                        +(lambda_w_est_m+lambda_yr_w_est_m*trend[year])
                        *delta_est_m
                        +(lambda_w_est_c+lambda_yr_w_est_c*trend[year])
                        *delta_est_c
                    )
                ) + 2*beta2_one_est
                part2_f = (
                    lambda_w_est_f+lambda_yr_w_est_f*trend[year]-1
                )*delta_est_f + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_one_est*w)/3
                )
                part2_m = (
                    lambda_w_est_m+lambda_yr_w_est_m*trend[year]-1
                )*delta_est_m + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_one_est*w)/3
                )
                part2_c = (
                    lambda_w_est_c+lambda_yr_w_est_c*trend[year]-1
                )*delta_est_c + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_one_est*w)/3
                )
            part1_wgh = 1/(2*3*beta2_one_est*w)
            part2_wgh = 1/(2*beta2_one_est*w*w)
            marginal_f = part1_wgh*part1 - part2_wgh*part2_f
            marginal_m = part1_wgh*part1 - part2_wgh*part2_m
            marginal_c = part1_wgh*part1 - part2_wgh*part2_c
            marginal[key] = {'female': marginal_f, 
                             'male': marginal_m, 
                             'children': marginal_c}
        else:
            delta_est_f = np.float64(values[0]['delta_est'])
            delta_est_m = np.float64(values[1]['delta_est'])
            delta_est_c = np.float64(values[2]['delta_est'])
            lambda_w_est_f = np.float64(values[0]['lambda_w_est'])
            lambda_w_est_m = np.float64(values[1]['lambda_w_est'])
            lambda_w_est_c = np.float64(values[2]['lambda_w_est'])
            if year == '2004':
                delta_est_f = np.exp(delta_est_f + lambda_w_est_f*w_demean)
                delta_est_m = np.exp(delta_est_m + lambda_w_est_m*w_demean)
                delta_est_c = np.exp(delta_est_c + lambda_w_est_c*w_demean)
                part1 = (
                    (1/w)*(
                        lambda_w_est_f*delta_est_f
                        +lambda_w_est_m*delta_est_m
                        +lambda_w_est_c*delta_est_c
                    )
                ) + 2*beta2_two_est
                part2_f = (lambda_w_est_f-1)*delta_est_f + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_two_est*w)/3
                )
                part2_m = (lambda_w_est_m-1)*delta_est_m + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_two_est*w)/3
                )
                part2_c = (lambda_w_est_c-1)*delta_est_c + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_two_est*w)/3
                )
            else:
                lambda_yr_est_f = np.float64(values[0]['lambda_yr_est'])
                lambda_yr_est_m = np.float64(values[1]['lambda_yr_est'])
                lambda_yr_est_c = np.float64(values[2]['lambda_yr_est'])
                lambda_yr_w_est_f = np.float64(values[0]['lambda_yr_w_est'])
                lambda_yr_w_est_m = np.float64(values[1]['lambda_yr_w_est'])
                lambda_yr_w_est_c = np.float64(values[2]['lambda_yr_w_est'])
                delta_est_f = np.exp(delta_est_f + lambda_yr_est_f 
                                    + lambda_w_est_f*w_demean 
                                    + lambda_yr_w_est_f*w_demean*trend[year])
                delta_est_m = np.exp(delta_est_m + lambda_yr_est_m 
                                    + lambda_w_est_m*w_demean 
                                    + lambda_yr_w_est_m*w_demean*trend[year])
                delta_est_c = np.exp(delta_est_c + lambda_yr_est_c 
                                    + lambda_w_est_c*w_demean 
                                    + lambda_yr_w_est_c*w_demean*trend[year])
                part1 = (
                    (1/w)*(
                        (lambda_w_est_f+lambda_yr_w_est_f*trend[year])
                        *delta_est_f
                        +(lambda_w_est_m+lambda_yr_w_est_m*trend[year])
                        *delta_est_m
                        +(lambda_w_est_c+lambda_yr_w_est_c*trend[year])
                        *delta_est_c
                    )
                ) + 2*beta2_two_est
                part2_f = (
                    lambda_w_est_f+lambda_yr_w_est_f*trend[year]-1
                )*delta_est_f + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_two_est*w)/3
                )
                part2_m = (
                    lambda_w_est_m+lambda_yr_w_est_m*trend[year]-1
                )*delta_est_m + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_two_est*w)/3
                )
                part2_c = (
                    lambda_w_est_c+lambda_yr_w_est_c*trend[year]-1
                )*delta_est_c + (
                    (delta_est_f+delta_est_m+delta_est_c+2*beta2_two_est*w)/3
                )
            part1_wgh = 1/(2*3*beta2_one_est*w)
            part2_wgh = 1/(2*beta2_one_est*w*w)
            marginal_f = part1_wgh*part1 - part2_wgh*part2_f
            marginal_m = part1_wgh*part1 - part2_wgh*part2_m
            marginal_c = part1_wgh*part1 - part2_wgh*part2_c
            marginal[key] = {'female': marginal_f, 
                             'male': marginal_m, 
                             'children': marginal_c}
            
    return marginal

# Computation starts --------------------
if __name__ == '__main__':
    # Parameters --------------------
    n_attr_f = 5
    n_attr_m = 5
    n_attr_c = 5
    batch_size = 1024
    n_imposters = 50
    max_q = 3
    device = torch.device("cpu")
    torch.manual_seed(123)

    # Hyperparameters --------------------
    lr = 1e-3
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    patience = 30
    max_epoch = 700

    # Load data --------------------
    os.chdir('/Users/apple/Desktop/Research/resource_shares/data')
    consumption_df = pd.read_csv('consumption_10.csv')
    upc_df = pd.read_csv('upc_10.csv')

    # Assign each UPC a unique index --------------------
    upc_mapping = pd.DataFrame(columns=['upc', 'item'])

    for gd in ['female', 'male', 'children']:
        upc_mapping_new = pd.DataFrame({
            'upc': upc_df.loc[upc_df['gender'] == gd]['upc'].unique(),
            'item': range(
                len(upc_df.loc[upc_df['gender'] == gd]['upc'].unique()))
        })
        upc_mapping = pd.concat([upc_mapping, upc_mapping_new])

    upc_mapping = upc_mapping.set_index('upc')
    upc_df = upc_df.join(upc_mapping, on='upc')
    upc_df['row'] = upc_df['panel_year'] - 2004

    # Create a price array for each gender --------------------
    row_f = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'female']['row'].values, 
        dtype=torch.long, device=device)
    col_f = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'female']['item'].values.astype(int), 
        dtype=torch.long, device=device)
    val_f = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'female']['price'].values, 
        dtype=torch.float32, device=device)
    n_upc_f = col_f.max()+1
    price_f = torch.zeros((17, n_upc_f), dtype=torch.float32, device=device)
    price_f[row_f, col_f] = val_f

    row_m = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'male']['row'].values, 
        dtype=torch.long, device=device)
    col_m = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'male']['item'].values.astype(int), 
        dtype=torch.long, device=device)
    val_m = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'male']['price'].values, 
        dtype=torch.float32, device=device)
    n_upc_m = col_m.max()+1
    price_m = torch.zeros((17, n_upc_m), dtype=torch.float32, device=device)
    price_m[row_m, col_m] = val_m

    row_c = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'children']['row'].values, 
        dtype=torch.long, device=device)
    col_c = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'children']['item'].values.astype(int), 
        dtype=torch.long, device=device)
    val_c = torch.tensor(
        upc_df.loc[upc_df['gender'] == 'children']['price'].values, 
        dtype=torch.float32, device=device)
    n_upc_c = col_c.max()+1
    price_c = torch.zeros((17, n_upc_c), dtype=torch.float32, device=device)
    price_c[row_c, col_c] = val_c

    # Identify candidate UPCs of each year --------------------
    candidate_f = {}
    candidate_m = {}
    candidate_c = {}

    for i in range(17):
        candidate_f[i] = torch.nonzero(price_f[i], as_tuple=True)[0]
        candidate_m[i] = torch.nonzero(price_m[i], as_tuple=True)[0]
        candidate_c[i] = torch.nonzero(price_c[i], as_tuple=True)[0]

    # Create group identifier --------------------
    gender_idx = {'female': 0, 'male': 1, 'children': 2}
    consumption_df['gender_idx'] = consumption_df['gender'].map(gender_idx)

    type_idx = {'couples': 0, 'one_child': 1, 'two_children': 2}
    consumption_df['type_idx'] = consumption_df['type'].map(type_idx)

    trend = {}

    for i, yr in enumerate(range(2004, 2021)):
        trend[yr] = i / (len(range(2004, 2021))-1)

    consumption_df['trend'] = consumption_df['panel_year'].map(trend)

    consumption_df['delta_group'] = (
        consumption_df['gender_idx'].astype(str) + '_' +
        consumption_df['panel_year'].astype(str) + '_' +
        consumption_df['type_idx'].astype(str)
    )
    delta_group = sorted(consumption_df['delta_group'].unique())
    delta_idx = {gp: idx for idx, gp in enumerate(delta_group)}
    delta_idx_2004 = torch.tensor(
        [val for key, val in delta_idx.items() if '_2004_' in key], 
        device=device)
    consumption_df['delta_idx'] = consumption_df['delta_group'].map(delta_idx)

    log_income = {13: np.log(1875).item(),
                  15: np.log(2292).item(),
                  16: np.log(2708).item(),
                  17: np.log(3125).item(),
                  18: np.log(3542).item(),
                  19: np.log(3958).item(),
                  21: np.log(4583).item(),
                  23: np.log(5417).item()}
    consumption_df['log_income'] = consumption_df[
        'household_income'].map(log_income)
    log_w_mean = consumption_df['log_income'].mean()
    consumption_df['log_income'] = consumption_df['log_income'] - log_w_mean

    consumption_df = consumption_df.sort_values(
        'panel_year').reset_index(drop=True)
    consumption_f = consumption_df.loc[
        consumption_df['gender'] == 'female'].reset_index(drop=True)
    consumption_m = consumption_df.loc[
        consumption_df['gender'] == 'male'].reset_index(drop=True)
    consumption_c = consumption_df.loc[
        consumption_df['gender'] == 'children'].reset_index(drop=True)

    nobs_f = len(consumption_f)
    nobs_m = len(consumption_m)
    nobs_c = len(consumption_c)

    yr_idx_f = consumption_f['panel_year'].values - 2004
    yr_idx_m = consumption_m['panel_year'].values - 2004
    yr_idx_c = consumption_c['panel_year'].values - 2004

    tp_idx_f = torch.tensor(
        consumption_f['type_idx'].values, dtype=torch.long, device=device)
    tp_idx_m = torch.tensor(
        consumption_m['type_idx'].values, dtype=torch.long, device=device)
    tp_idx_c = torch.tensor(
        consumption_c['type_idx'].values, dtype=torch.long, device=device)

    delta_idx_f = torch.tensor(
        consumption_f['delta_idx'].values, dtype=torch.long, device=device)
    delta_idx_m = torch.tensor(
        consumption_m['delta_idx'].values, dtype=torch.long, device=device)
    delta_idx_c = torch.tensor(
        consumption_c['delta_idx'].values, dtype=torch.long, device=device)

    log_w_f = torch.tensor(
        consumption_f['log_income'].values, dtype=torch.float32, device=device)
    log_w_m = torch.tensor(
        consumption_m['log_income'].values, dtype=torch.float32, device=device)
    log_w_c = torch.tensor(
        consumption_c['log_income'].values, dtype=torch.float32, device=device)
    
    trend_f = torch.tensor(
        consumption_f['trend'].values, dtype=torch.float32, device=device)
    trend_m = torch.tensor(
        consumption_m['trend'].values, dtype=torch.float32, device=device)
    trend_c = torch.tensor(
        consumption_c['trend'].values, dtype=torch.float32, device=device)

    # Create demand and total spending tensors --------------------
    # Female
    upc_to_idx_f = dict(zip(upc_df.loc[upc_df['gender'] == 'female']['upc'], 
                            upc_df.loc[upc_df['gender'] == 'female']['item']))
    row_f = []
    col_f = []
    q_val_f = []
    p_val_f = []

    for i, (upc_str, qty_str, pcs_str) in enumerate(
        zip(consumption_f['upc'], 
            consumption_f['quantity'],
            consumption_f['price'])
    ):
        upcs = upc_str.split(', ')
        quantities = map(float, qty_str.split(', '))
        prices = map(float, pcs_str.split(', '))

        for upc, qty, pcs in zip(upcs, quantities, prices):
            row_f.append(i)
            col_f.append(upc_to_idx_f[upc])
            q_val_f.append(qty)
            p_val_f.append(pcs)

    row_f = torch.tensor(row_f, dtype=torch.long, device=device)
    col_f = torch.tensor(col_f, dtype=torch.long, device=device)
    q_val_f = torch.tensor(q_val_f, dtype=torch.float32, device=device)
    p_val_f = torch.tensor(p_val_f, dtype=torch.float32, device=device)

    # Male
    upc_to_idx_m = dict(zip(upc_df.loc[upc_df['gender'] == 'male']['upc'], 
                            upc_df.loc[upc_df['gender'] == 'male']['item']))
    row_m = []
    col_m = []
    q_val_m = []
    p_val_m = []

    for i, (upc_str, qty_str, pcs_str) in enumerate(
        zip(consumption_m['upc'], 
            consumption_m['quantity'],
            consumption_m['price'])
    ):
        upcs = upc_str.split(', ')
        quantities = map(float, qty_str.split(', '))
        prices = map(float, pcs_str.split(', '))

        for upc, qty, pcs in zip(upcs, quantities, prices):
            row_m.append(i)
            col_m.append(upc_to_idx_m[upc])
            q_val_m.append(qty)
            p_val_m.append(pcs)

    row_m = torch.tensor(row_m, dtype=torch.long, device=device)
    col_m = torch.tensor(col_m, dtype=torch.long, device=device)
    q_val_m = torch.tensor(q_val_m, dtype=torch.float32, device=device)
    p_val_m = torch.tensor(p_val_m, dtype=torch.float32, device=device)

    # Children
    upc_to_idx_c = dict(zip(upc_df.loc[upc_df['gender'] == 'children']['upc'], 
                            upc_df.loc[upc_df['gender'] == 'children']['item']))
    row_c = []
    col_c = []
    q_val_c = []
    p_val_c = []

    for i, (upc_str, qty_str, pcs_str) in enumerate(
        zip(consumption_c['upc'], 
            consumption_c['quantity'],
            consumption_c['price'])
    ):
        upcs = upc_str.split(', ')
        quantities = map(float, qty_str.split(', '))
        prices = map(float, pcs_str.split(', '))

        for upc, qty, pcs in zip(upcs, quantities, prices):
            row_c.append(i)
            col_c.append(upc_to_idx_c[upc])
            q_val_c.append(qty)
            p_val_c.append(pcs)

    row_c = torch.tensor(row_c, dtype=torch.long, device=device)
    col_c = torch.tensor(col_c, dtype=torch.long, device=device)
    q_val_c = torch.tensor(q_val_c, dtype=torch.float32, device=device)
    p_val_c = torch.tensor(p_val_c, dtype=torch.float32, device=device)

    del consumption_df, consumption_f, consumption_m, consumption_c
    gc.collect()

    # Create imposters --------------------
    # Female
    row_im_f = row_f.repeat(n_imposters)
    col_im_f = col_f.repeat(n_imposters)
    q_im_val_f = q_val_f.repeat(n_imposters)
    p_im_val_f = p_val_f.repeat(n_imposters)
    for i in range(nobs_f):
        row_idx = (row_f==i).nonzero(as_tuple=True)[0]
        item_bought = col_f[row_idx]
        candidate = candidate_f[yr_idx_f[i]]
        candidate = candidate[~torch.isin(candidate, item_bought)]
        if len(row_idx) > 1:
            idx_del = row_idx[
                torch.randint(high=len(row_idx), size=(1, ), device=device)]
            idx_del = torch.tensor(
                [x*len(col_f)+idx_del for x in range(n_imposters)], 
                device=device)
        else:
            idx_del = torch.tensor(
                [x*len(col_f)+row_idx for x in range(n_imposters)], 
                device=device)
        selected_idx = torch.randperm(len(candidate), 
                                      device=device)[:n_imposters]
        selected = candidate[selected_idx]
        quantity = torch.randint(
            1, max_q+1, size=(n_imposters, ), device=device)
        price = price_f[yr_idx_f[i], selected]
        col_im_f[idx_del] = selected
        q_im_val_f[idx_del] = quantity.to(dtype=torch.float32)
        p_im_val_f[idx_del] = price

    row_im_f = torch.cat((row_f, row_im_f))
    col_im_f = torch.cat((col_f, col_im_f))
    q_im_val_f = torch.cat((q_val_f, q_im_val_f))
    p_im_val_f = torch.cat((p_val_f, p_im_val_f))
    pq_im_val_f = torch.split(q_im_val_f*p_im_val_f, len(row_f))
    imp_idx_f = torch.arange(
        n_imposters+1, device=device).repeat_interleave(len(row_f))

    pq_im_f = torch.zeros((nobs_f, n_imposters+1), device=device)
    for i in range(n_imposters+1):
        pq_im_f[:, i] = pq_im_f[:, i].scatter_add(0, row_f, pq_im_val_f[i])

    # Male
    row_im_m = row_m.repeat(n_imposters)
    col_im_m = col_m.repeat(n_imposters)
    q_im_val_m = q_val_m.repeat(n_imposters)
    p_im_val_m = p_val_m.repeat(n_imposters)
    for i in range(nobs_m):
        row_idx = (row_m==i).nonzero(as_tuple=True)[0]
        item_bought = col_m[row_idx]
        candidate = candidate_m[yr_idx_m[i]]
        candidate = candidate[~torch.isin(candidate, item_bought)]
        if len(row_idx) > 1:
            idx_del = row_idx[
                torch.randint(high=len(row_idx), size=(1, ), device=device)]
            idx_del = torch.tensor(
                [x*len(col_m)+idx_del for x in range(n_imposters)], 
                device=device)
        else:
            idx_del = torch.tensor(
                [x*len(col_m)+row_idx for x in range(n_imposters)], 
                device=device)
        selected_idx = torch.randperm(len(candidate), 
                                      device=device)[:n_imposters]
        selected = candidate[selected_idx]
        quantity = torch.randint(
            1, max_q+1, size=(n_imposters, ), device=device)
        price = price_m[yr_idx_m[i], selected]
        col_im_m[idx_del] = selected
        q_im_val_m[idx_del] = quantity.to(dtype=torch.float32)
        p_im_val_m[idx_del] = price

    row_im_m = torch.cat((row_m, row_im_m))
    col_im_m = torch.cat((col_m, col_im_m))
    q_im_val_m = torch.cat((q_val_m, q_im_val_m))
    p_im_val_m = torch.cat((p_val_m, p_im_val_m))
    pq_im_val_m = torch.split(q_im_val_m*p_im_val_m, len(row_m))
    imp_idx_m = torch.arange(
        n_imposters+1, device=device).repeat_interleave(len(row_m))

    pq_im_m = torch.zeros((nobs_m, n_imposters+1), device=device)
    for i in range(n_imposters+1):
        pq_im_m[:, i] = pq_im_m[:, i].scatter_add(0, row_m, pq_im_val_m[i])

    # Children
    row_im_c = row_c.repeat(n_imposters)
    col_im_c = col_c.repeat(n_imposters)
    q_im_val_c = q_val_c.repeat(n_imposters)
    p_im_val_c = p_val_c.repeat(n_imposters)
    for i in range(nobs_c):
        row_idx = (row_c==i).nonzero(as_tuple=True)[0]
        item_bought = col_c[row_idx]
        candidate = candidate_c[yr_idx_c[i]]
        candidate = candidate[~torch.isin(candidate, item_bought)]
        if len(row_idx) > 1:
            idx_del = row_idx[
                torch.randint(high=len(row_idx), size=(1, ), device=device)]
            idx_del = torch.tensor(
                [x*len(col_c)+idx_del for x in range(n_imposters)], 
                device=device)
        else:
            idx_del = torch.tensor(
                [x*len(col_c)+row_idx for x in range(n_imposters)], 
                device=device)
        selected_idx = torch.randperm(len(candidate), 
                                      device=device)[:n_imposters]
        selected = candidate[selected_idx]
        quantity = torch.randint(
            1, max_q+1, size=(n_imposters, ), device=device)
        price = price_c[yr_idx_c[i], selected]
        col_im_c[idx_del] = selected
        q_im_val_c[idx_del] = quantity.to(dtype=torch.float32)
        p_im_val_c[idx_del] = price

    row_im_c = torch.cat((row_c, row_im_c))
    col_im_c = torch.cat((col_c, col_im_c))
    q_im_val_c = torch.cat((q_val_c, q_im_val_c))
    p_im_val_c = torch.cat((p_val_c, p_im_val_c))
    pq_im_val_c = torch.split(q_im_val_c*p_im_val_c, len(row_c))
    imp_idx_c = torch.arange(
        n_imposters+1, device=device).repeat_interleave(len(row_c))

    pq_im_c = torch.zeros((nobs_c, n_imposters+1), device=device)
    for i in range(n_imposters+1):
        pq_im_c[:, i] = pq_im_c[:, i].scatter_add(0, row_c, pq_im_val_c[i])

    # Split observations into batches --------------------
    nobs = nobs_f + nobs_m + nobs_c
    batch_size_f = round(batch_size * nobs_f/nobs)
    batch_size_m = round(batch_size * nobs_m/nobs)
    batch_size_c = round(batch_size * nobs_c/nobs)

    obs_idx_f = tuple(
        torch.sort(x)[0] 
        for x in torch.randperm(nobs_f, device=device).split(batch_size_f)
    )
    obs_idx_m = tuple(
        torch.sort(x)[0] 
        for x in torch.randperm(nobs_m, device=device).split(batch_size_m)
    )
    obs_idx_c = tuple(
        torch.sort(x)[0] 
        for x in torch.randperm(nobs_c, device=device).split(batch_size_c)
    )
    n_batch = len(obs_idx_f) - 1

    # Validation data --------------------
    valdn_idx = n_batch

    idx_valdn_f = obs_idx_f[valdn_idx]
    f_valdn = torch.isin(row_im_f, idx_valdn_f)
    q_valdn_f = torch.sparse_coo_tensor(
        indices=torch.stack((
                torch.searchsorted(idx_valdn_f, row_im_f[f_valdn]), 
                col_im_f[f_valdn], 
                imp_idx_f[f_valdn])),
        values=q_im_val_f[f_valdn],
        size=(len(idx_valdn_f), n_upc_f, n_imposters+1)
    )
    pq_valdn_f = pq_im_f[idx_valdn_f]
    tp_valdn_f = tp_idx_f[idx_valdn_f]
    delta_valdn_f = delta_idx_f[idx_valdn_f]
    log_w_valdn_f = log_w_f[idx_valdn_f]
    trend_valdn_f = trend_f[idx_valdn_f]
    yr_valdn_f = yr_idx_f[idx_valdn_f]

    idx_valdn_m = obs_idx_m[valdn_idx]
    m_valdn = torch.isin(row_im_m, idx_valdn_m)
    q_valdn_m = torch.sparse_coo_tensor(
        indices=torch.stack((
                torch.searchsorted(idx_valdn_m, row_im_m[m_valdn]), 
                col_im_m[m_valdn], 
                imp_idx_m[m_valdn])),
        values=q_im_val_m[m_valdn],
        size=(len(idx_valdn_m), n_upc_m, n_imposters+1)
    )
    pq_valdn_m = pq_im_m[idx_valdn_m]
    tp_valdn_m = tp_idx_m[idx_valdn_m]
    delta_valdn_m = delta_idx_m[idx_valdn_m]
    log_w_valdn_m = log_w_m[idx_valdn_m]
    trend_valdn_m = trend_m[idx_valdn_m]
    yr_valdn_m = yr_idx_m[idx_valdn_m]

    idx_valdn_c = obs_idx_c[valdn_idx]
    c_valdn = torch.isin(row_im_c, idx_valdn_c)
    q_valdn_c = torch.sparse_coo_tensor(
        indices=torch.stack((
                torch.searchsorted(idx_valdn_c, row_im_c[c_valdn]), 
                col_im_c[c_valdn], 
                imp_idx_c[c_valdn])),
        values=q_im_val_c[c_valdn],
        size=(len(idx_valdn_c), n_upc_c, n_imposters+1)
    )
    pq_valdn_c = pq_im_c[idx_valdn_c]
    tp_valdn_c = tp_idx_c[idx_valdn_c]
    delta_valdn_c = delta_idx_c[idx_valdn_c]
    log_w_valdn_c = log_w_c[idx_valdn_c]
    trend_valdn_c = trend_c[idx_valdn_c]
    yr_valdn_c = yr_idx_c[idx_valdn_c]

    # Training data --------------------
    idx_train_f = (~torch.isin(torch.arange(nobs_f, device=device), 
                            idx_valdn_f)).nonzero(as_tuple=True)[0]
    idx_train_m = (~torch.isin(torch.arange(nobs_m, device=device), 
                            idx_valdn_m)).nonzero(as_tuple=True)[0]
    idx_train_c = (~torch.isin(torch.arange(nobs_c, device=device), 
                            idx_valdn_c)).nonzero(as_tuple=True)[0]

    len_train_f = len(idx_train_f)
    len_train_m = len(idx_train_m)
    len_train_c = len(idx_train_c)

    # Initialize parameters --------------------
    n_upc_f = price_f.shape[1]
    n_upc_m = price_m.shape[1]
    n_upc_c = price_c.shape[1]

    n_attr = n_attr_f + n_attr_m + n_attr_c

    init_gama_f = torch.empty(
        (n_attr_f, n_upc_f), device=device).normal_(0, 1/n_attr)
    init_gama_m = torch.empty(
        (n_attr_m, n_upc_m), device=device).normal_(0, 1/n_attr)
    init_gama_c = torch.empty(
        (n_attr_c, n_upc_c), device=device).normal_(0, 1/n_attr)

    init_alfa_cp_f = torch.empty((n_attr_f, ), 
                                 device=device).normal_(0, 1/n_attr)
    init_alfa_cp_m = torch.empty((n_attr_m, ), 
                                 device=device).normal_(0, 1/n_attr)

    init_alfa_one_f = torch.empty((n_attr_f, ), 
                                  device=device).normal_(0, 1/n_attr)
    init_alfa_one_m = torch.empty((n_attr_m, ), 
                                  device=device).normal_(0, 1/n_attr)
    init_alfa_one_c = torch.empty((n_attr_c, ), 
                                  device=device).normal_(0, 1/n_attr)

    init_alfa_two_f = torch.empty((n_attr_f, ), 
                                  device=device).normal_(0, 1/n_attr)
    init_alfa_two_m = torch.empty((n_attr_m, ), 
                                  device=device).normal_(0, 1/n_attr)
    init_alfa_two_c = torch.empty((n_attr_c, ), 
                                  device=device).normal_(0, 1/n_attr)

    init_delta = torch.log(torch.empty((8, ), device=device).exponential_())
    init_beta2 = torch.log(torch.empty((3, ), device=device).uniform_(0.1, 0.2))

    init_lambda_w = torch.zeros(
        (8, ), dtype=torch.float32, device=device)
    init_lambda_yr = torch.zeros(
        (len(delta_idx)-len(delta_idx_2004), ), dtype=torch.float32, 
        device=device)
    init_lambda_yr_w = torch.zeros(
        (8, ), dtype=torch.float32, device=device)
    
    init_alfa_yr_f = torch.zeros(
        (16, n_attr_f), dtype=torch.float32, device=device)
    init_alfa_yr_m = torch.zeros(
        (16, n_attr_m), dtype=torch.float32, device=device)
    init_alfa_yr_c = torch.zeros(
        (16, n_attr_c), dtype=torch.float32, device=device)

    init = torch.cat((
        init_gama_f.flatten(), 
        init_gama_m.flatten(),
        init_gama_c.flatten(),
        init_alfa_cp_f,
        init_alfa_cp_m,
        init_alfa_one_f,
        init_alfa_one_m,
        init_alfa_one_c,
        init_alfa_two_f,
        init_alfa_two_m,
        init_alfa_two_c,
        init_delta,
        init_beta2,
        init_lambda_w,
        init_lambda_yr,
        init_lambda_yr_w,
        init_alfa_yr_f.flatten(),
        init_alfa_yr_m.flatten(),
        init_alfa_yr_c.flatten())).unsqueeze(0)

    param_idx = torch.cumsum(
        torch.tensor([
            0, n_attr_f*n_upc_f, n_attr_m*n_upc_m, n_attr_c*n_upc_c, n_attr_f, 
            n_attr_m, n_attr_f, n_attr_m, n_attr_c, n_attr_f, n_attr_m, 
            n_attr_c, 8, 3, 8, len(delta_idx)-len(delta_idx_2004), 8, 
            n_attr_f*16, n_attr_m*16, n_attr_c*16
        ], device=device), dim=0)

    # Gradient descent --------------------
    # Initialize algorithm
    m = torch.zeros_like(init, device=device)
    v = torch.zeros_like(init, device=device)
    t = 0
    batch = 0
    epoch = 1
    best_valdn_loss = torch.tensor([float('inf')], device=device)
    no_improvement_counter = 0

    # Store the results
    loss_t = torch.zeros(max_epoch*n_batch, device=device)
    valdn_t = torch.zeros(max_epoch, device=device)
    improvement_t = torch.zeros(max_epoch, device=device)
    grad_norm = torch.zeros(max_epoch*n_batch, device=device)

    # Model parameters
    params = init.clone().requires_grad_()
    
    while epoch <= max_epoch:
        idx_selected_f = obs_idx_f[batch]
        f_batch = torch.isin(row_im_f, idx_selected_f)
        q_batch_f = torch.sparse_coo_tensor(
            indices=torch.stack((
                torch.searchsorted(idx_selected_f, row_im_f[f_batch]), 
                col_im_f[f_batch], 
                imp_idx_f[f_batch])),
            values=q_im_val_f[f_batch],
            size=(batch_size_f, n_upc_f, n_imposters+1)
        )
        pq_batch_f = pq_im_f[idx_selected_f]
        tp_batch_f = tp_idx_f[idx_selected_f]
        delta_batch_f = delta_idx_f[idx_selected_f]
        log_w_batch_f = log_w_f[idx_selected_f]
        trend_batch_f = trend_f[idx_selected_f]
        yr_batch_f = yr_idx_f[idx_selected_f]

        idx_selected_m = obs_idx_m[batch]
        m_batch = torch.isin(row_im_m, idx_selected_m)
        q_batch_m = torch.sparse_coo_tensor(
            indices=torch.stack((
                torch.searchsorted(idx_selected_m, row_im_m[m_batch]), 
                col_im_m[m_batch], 
                imp_idx_m[m_batch])),
            values=q_im_val_m[m_batch],
            size=(batch_size_m, n_upc_m, n_imposters+1)
        )
        pq_batch_m = pq_im_m[idx_selected_m]
        tp_batch_m = tp_idx_m[idx_selected_m]
        delta_batch_m = delta_idx_m[idx_selected_m]
        log_w_batch_m = log_w_m[idx_selected_m]
        trend_batch_m = trend_m[idx_selected_m]
        yr_batch_m = yr_idx_m[idx_selected_m]

        idx_selected_c = obs_idx_c[batch]
        c_batch = torch.isin(row_im_c, idx_selected_c)
        q_batch_c = torch.sparse_coo_tensor(
            indices=torch.stack((
                torch.searchsorted(idx_selected_c, row_im_c[c_batch]), 
                col_im_c[c_batch], 
                imp_idx_c[c_batch])),
            values=q_im_val_c[c_batch],
            size=(batch_size_c, n_upc_c, n_imposters+1)
        )
        pq_batch_c = pq_im_c[idx_selected_c]
        tp_batch_c = tp_idx_c[idx_selected_c]
        delta_batch_c = delta_idx_c[idx_selected_c]
        log_w_batch_c = log_w_c[idx_selected_c]
        trend_batch_c = trend_c[idx_selected_c]
        yr_batch_c = yr_idx_c[idx_selected_c]

        loss = mle(params, q_batch_f, q_batch_m, q_batch_c, 
                   pq_batch_f, pq_batch_m, pq_batch_c, 
                   tp_batch_f, tp_batch_m, tp_batch_c, 
                   log_w_batch_f, log_w_batch_m, log_w_batch_c,
                   delta_batch_f, delta_batch_m, delta_batch_c,
                   trend_batch_f, trend_batch_m, trend_batch_c,
                   yr_batch_f, yr_batch_m, yr_batch_c,
                   param_idx, n_attr_f, n_attr_m, n_attr_c,
                   n_upc_f, n_upc_m, n_upc_c, delta_idx_2004, device)
        loss_t[t] = loss.item()
        loss.backward()

        with torch.no_grad():
            gradient = params.grad
            grad_norm[t] = torch.linalg.norm(gradient)
            t += 1
            m = beta_1*m + (1-beta_1)*gradient
            v = beta_2*v + (1-beta_2)*(gradient**2)
            m_hat = m / (1-beta_1**t)
            v_hat = v / (1-beta_2**t)
            params -= lr*m_hat / (torch.sqrt(v_hat)+epsilon)
            if (epoch == max_epoch) & ((batch+1) == n_batch):
                print('Maximum number of epochs reached.')
                break
        
        params.grad.zero_()
        batch = (batch+1) % n_batch
        if batch == 0:
            if epoch % 50 == 0:
                print(f'Epoch {epoch} done. Loss={loss_t[t-1]:.5f}.', 
                      f'Grad={grad_norm[t-1]:.5f}.')

            # Validation
            with torch.no_grad():
                valdn_loss = mle(
                    params, q_valdn_f, q_valdn_m, q_valdn_c, 
                    pq_valdn_f, pq_valdn_m, pq_valdn_c, 
                    tp_valdn_f, tp_valdn_m, tp_valdn_c, 
                    log_w_valdn_f, log_w_valdn_m, log_w_valdn_c,
                    delta_valdn_f, delta_valdn_m, delta_valdn_c,
                    trend_valdn_f, trend_valdn_m, trend_valdn_c,
                    yr_valdn_f, yr_valdn_m, yr_valdn_c,
                    param_idx, n_attr_f, n_attr_m, n_attr_c,
                    n_upc_f, n_upc_m, n_upc_c, delta_idx_2004, device).item()
                valdn_t[epoch-1] = valdn_loss
                if valdn_loss < best_valdn_loss:
                    best_valdn_loss = valdn_loss
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
                improvement_t[epoch-1] = no_improvement_counter
                if no_improvement_counter == patience:
                    print(f'Early stopping triggered at the {epoch}-th epoch.')
                    break

            epoch += 1

            # Generate new imposters: female
            col_im_f = col_f.repeat(n_imposters)
            q_im_val_f = q_val_f.repeat(n_imposters)
            p_im_val_f = p_val_f.repeat(n_imposters)
            for i in range(nobs_f):
                row_idx = (row_f==i).nonzero(as_tuple=True)[0]
                item_bought = col_f[row_idx]
                candidate = candidate_f[yr_idx_f[i]]
                candidate = candidate[~torch.isin(candidate, item_bought)]
                if len(row_idx) > 1:
                    idx_del = row_idx[
                        torch.randint(high=len(row_idx), size=(1, ), 
                                      device=device)]
                    idx_del = torch.tensor(
                        [x*len(col_f)+idx_del for x in range(n_imposters)], 
                        device=device)
                else:
                    idx_del = torch.tensor(
                        [x*len(col_f)+row_idx for x in range(n_imposters)], 
                        device=device)
                selected_idx = torch.randperm(
                    len(candidate), device=device)[:n_imposters]
                selected = candidate[selected_idx]
                quantity = torch.randint(
                    1, max_q+1, size=(n_imposters, ), device=device)
                price = price_f[yr_idx_f[i], selected]
                col_im_f[idx_del] = selected
                q_im_val_f[idx_del] = quantity.to(dtype=torch.float32)
                p_im_val_f[idx_del] = price

            col_im_f = torch.cat((col_f, col_im_f))
            q_im_val_f = torch.cat((q_val_f, q_im_val_f))
            p_im_val_f = torch.cat((p_val_f, p_im_val_f))
            pq_im_val_f = torch.split(q_im_val_f*p_im_val_f, len(row_f))

            pq_im_f = torch.zeros((nobs_f, n_imposters+1), device=device)
            for i in range(n_imposters+1):
                pq_im_f[:, i] = pq_im_f[:, i].scatter_add(
                    0, row_f, pq_im_val_f[i])

            # Generate new imposters: male
            col_im_m = col_m.repeat(n_imposters)
            q_im_val_m = q_val_m.repeat(n_imposters)
            p_im_val_m = p_val_m.repeat(n_imposters)
            for i in range(nobs_m):
                row_idx = (row_m==i).nonzero(as_tuple=True)[0]
                item_bought = col_m[row_idx]
                candidate = candidate_m[yr_idx_m[i]]
                candidate = candidate[~torch.isin(candidate, item_bought)]
                if len(row_idx) > 1:
                    idx_del = row_idx[
                        torch.randint(high=len(row_idx), size=(1, ), 
                                      device=device)]
                    idx_del = torch.tensor(
                        [x*len(col_m)+idx_del for x in range(n_imposters)], 
                        device=device)
                else:
                    idx_del = torch.tensor(
                        [x*len(col_m)+row_idx for x in range(n_imposters)], 
                        device=device)
                selected_idx = torch.randperm(
                    len(candidate), device=device)[:n_imposters]
                selected = candidate[selected_idx]
                quantity = torch.randint(
                    1, max_q+1, size=(n_imposters, ), device=device)
                price = price_m[yr_idx_m[i], selected]
                col_im_m[idx_del] = selected
                q_im_val_m[idx_del] = quantity.to(dtype=torch.float32)
                p_im_val_m[idx_del] = price

            col_im_m = torch.cat((col_m, col_im_m))
            q_im_val_m = torch.cat((q_val_m, q_im_val_m))
            p_im_val_m = torch.cat((p_val_m, p_im_val_m))
            pq_im_val_m = torch.split(q_im_val_m*p_im_val_m, len(row_m))

            pq_im_m = torch.zeros((nobs_m, n_imposters+1), device=device)
            for i in range(n_imposters+1):
                pq_im_m[:, i] = pq_im_m[:, i].scatter_add(
                    0, row_m, pq_im_val_m[i])

            # Generate new imposters: children
            col_im_c = col_c.repeat(n_imposters)
            q_im_val_c = q_val_c.repeat(n_imposters)
            p_im_val_c = p_val_c.repeat(n_imposters)
            for i in range(nobs_c):
                row_idx = (row_c==i).nonzero(as_tuple=True)[0]
                item_bought = col_c[row_idx]
                candidate = candidate_c[yr_idx_c[i]]
                candidate = candidate[~torch.isin(candidate, item_bought)]
                if len(row_idx) > 1:
                    idx_del = row_idx[
                        torch.randint(high=len(row_idx), size=(1, ), 
                                      device=device)]
                    idx_del = torch.tensor(
                        [x*len(col_c)+idx_del for x in range(n_imposters)], 
                        device=device)
                else:
                    idx_del = torch.tensor(
                        [x*len(col_c)+row_idx for x in range(n_imposters)], 
                        device=device)
                selected_idx = torch.randperm(
                    len(candidate), device=device)[:n_imposters]
                selected = candidate[selected_idx]
                quantity = torch.randint(
                    1, max_q+1, size=(n_imposters, ), device=device)
                price = price_c[yr_idx_c[i], selected]
                col_im_c[idx_del] = selected
                q_im_val_c[idx_del] = quantity.to(dtype=torch.float32)
                p_im_val_c[idx_del] = price

            col_im_c = torch.cat((col_c, col_im_c))
            q_im_val_c = torch.cat((q_val_c, q_im_val_c))
            p_im_val_c = torch.cat((p_val_c, p_im_val_c))
            pq_im_val_c = torch.split(q_im_val_c*p_im_val_c, len(row_c))

            pq_im_c = torch.zeros((nobs_c, n_imposters+1), device=device)
            for i in range(n_imposters+1):
                pq_im_c[:, i] = pq_im_c[:, i].scatter_add(
                    0, row_c, pq_im_val_c[i])

    # Calculate resource shares --------------------
    delta_est = params[:, range(param_idx[11], param_idx[12])].squeeze()
    lambda_w_est = params[:, range(param_idx[13], param_idx[14])].squeeze()
    lambda_yr_est = params[:, range(param_idx[14], param_idx[15])].squeeze()
    lambda_yr_w_est = params[
        :, range(param_idx[15], param_idx[16])].squeeze()
    beta2_cp_est = np.exp(params[:, param_idx[12]].item())
    beta2_one_est = np.exp(params[:, param_idx[12]+1].item())
    beta2_two_est = np.exp(params[:, param_idx[12]+2].item())

    (
        shares, avg_shares_tp, avg_shares_tp_w, unreasonable_shares
    ) = resource_share(delta_est, lambda_w_est, lambda_yr_est, lambda_yr_w_est, 
                       beta2_cp_est, beta2_one_est, beta2_two_est)

    # Tables --------------------
    types = ['0', '1', '2']
    income = ['0', '1', '2', '3', '4', '5', '6', '7']
    columns = pd.MultiIndex.from_tuples(
        [(t, subcol) for t in types for subcol in ['female', 'male', 'children'] 
        if t != '0' or subcol != 'children']
    )
    shares_table = pd.DataFrame(index=['Overall']+income, 
                                columns=columns, 
                                dtype=float)

    for t, results in avg_shares_tp.items():
        for subcol, value in results.items():
            shares_table.loc['Overall', (t, subcol)] = value

    for key, results in avg_shares_tp_w.items():
        t_type, income = key.split('_')[:2]
        for subcol, value in results.items():
            shares_table.loc[income, (t_type, subcol)] = value

    col_names = {'0': 'Couples', '1': 'One Child', '2': 'Two Children'}
    shares_table.columns = shares_table.columns.set_levels(
        [col_names.get(t, t) for t in shares_table.columns.levels[0]],
        level=0
    )
    row_names = {'0': '$22,500', 
                '1': '$27,500', 
                '2': '$32,500', 
                '3': '$37,500', 
                '4': '$42,500', 
                '5': '$47,500', 
                '6': '$55,000', 
                '7': '$65,000'}
    shares_table.rename(index=lambda x: row_names.get(x, x), inplace=True)

    print(shares_table.style.format('{:.4f}').to_latex())

    # Bootstrap Starts --------------------
    # Bootstrap hyperparameter
    n_bt = 199

    # Batch indices used in bootstrap --------------------
    bt_idx_f = tuple(torch.sort(x)[0] 
    for x in torch.randperm(len_train_f, device=device).split(batch_size_f))
    bt_idx_m = tuple(torch.sort(x)[0] 
    for x in torch.randperm(len_train_m, device=device).split(batch_size_m))
    bt_idx_c = tuple(torch.sort(x)[0] 
    for x in torch.randperm(len_train_c, device=device).split(batch_size_c))

    # Resampling process --------------------
    # Maximum type/year indices
    max_gp_f = torch.max(delta_idx_f)
    max_gp_m = torch.max(delta_idx_m)

    # Number of observations of each type/year in the training set
    num_gp_f = torch.bincount(delta_idx_f[idx_train_f])
    num_gp_m = torch.bincount(delta_idx_m[idx_train_m] - max_gp_f - 1)
    num_gp_c = torch.bincount(delta_idx_c[idx_train_c] - max_gp_m - 1)

    # Resampling
    idx_bt_all = tuple((
        torch.cat(tuple(
            idx_train_f[torch.where(delta_idx_f[idx_train_f] == i)[0]][
                torch.randint(num_gp_f[i], size=(num_gp_f[i], ), device=device)
            ].to(torch.int16) for i in range(len(num_gp_f))
        )),
        torch.cat(tuple(
            idx_train_m[
                torch.where(delta_idx_m[idx_train_m] == (i+max_gp_f+1))[0]][
                    torch.randint(
                        num_gp_m[i], size=(num_gp_m[i], ), device=device)
            ].to(torch.int16) for i in range(len(num_gp_m))
        )),
        torch.cat(tuple(
            idx_train_c[
                torch.where(delta_idx_c[idx_train_c] == (i+max_gp_m+1))[0]][
                    torch.randint(
                        num_gp_c[i], size=(num_gp_c[i], ), device=device)
            ].to(torch.int16) for i in range(len(num_gp_c))
        ))
    ) for _ in range(n_bt))

    # Build shared memory --------------------
    len_val_f = len(row_f)
    row_f_np = row_f.numpy().astype(np.int16)
    row_f_shm = shared_memory.SharedMemory(create=True, size=row_f_np.nbytes)
    shared_row_f = np.ndarray(
        (len_val_f, ), dtype=np.int16, buffer=row_f_shm.buf)
    np.copyto(shared_row_f, row_f_np)    
    col_f_np = col_f.numpy().astype(np.int16)
    col_f_shm = shared_memory.SharedMemory(create=True, size=col_f_np.nbytes)
    shared_col_f = np.ndarray(
        (len_val_f, ), dtype=np.int16, buffer=col_f_shm.buf)
    np.copyto(shared_col_f, col_f_np)
    q_val_f_np = q_val_f.numpy().astype(np.int8)
    q_val_f_shm = shared_memory.SharedMemory(
        create=True, size=q_val_f_np.nbytes)
    shared_q_val_f = np.ndarray(
        (len_val_f, ), dtype=np.int8, buffer=q_val_f_shm.buf)
    np.copyto(shared_q_val_f, q_val_f_np)
    p_val_f_np = p_val_f.numpy().astype(np.float32)
    p_val_f_shm = shared_memory.SharedMemory(
        create=True, size=p_val_f_np.nbytes)
    shared_p_val_f = np.ndarray(
        (len_val_f, ), dtype=np.float32, buffer=p_val_f_shm.buf)
    np.copyto(shared_p_val_f, p_val_f_np)
    tp_idx_f_np = tp_idx_f.numpy().astype(np.int8)
    tp_idx_f_shm = shared_memory.SharedMemory(
        create=True, size=tp_idx_f_np.nbytes)
    shared_tp_idx_f = np.ndarray(
        (nobs_f, ), dtype=np.int8, buffer=tp_idx_f_shm.buf)
    np.copyto(shared_tp_idx_f, tp_idx_f_np)
    delta_idx_f_np = delta_idx_f.numpy().astype(np.int16)
    delta_idx_f_shm = shared_memory.SharedMemory(
        create=True, size=delta_idx_f_np.nbytes)
    shared_delta_idx_f = np.ndarray(
        (nobs_f, ), dtype=np.int16, buffer=delta_idx_f_shm.buf)
    np.copyto(shared_delta_idx_f, delta_idx_f_np)
    log_w_f_np = log_w_f.numpy().astype(np.float32)
    log_w_f_shm = shared_memory.SharedMemory(
        create=True, size=log_w_f_np.nbytes)
    shared_log_w_f = np.ndarray(
        (nobs_f, ), dtype=np.float32, buffer=log_w_f_shm.buf)
    np.copyto(shared_log_w_f, log_w_f_np)
    trend_f_np = trend_f.numpy().astype(np.float32)
    trend_f_shm = shared_memory.SharedMemory(
        create=True, size=trend_f_np.nbytes)
    shared_trend_f = np.ndarray(
        (nobs_f, ), dtype=np.float32, buffer=trend_f_shm.buf)
    np.copyto(shared_trend_f, trend_f_np)
    yr_idx_f_np = yr_idx_f.astype(np.int8)
    yr_idx_f_shm = shared_memory.SharedMemory(
        create=True, size=yr_idx_f_np.nbytes)
    shared_yr_idx_f = np.ndarray(
        (nobs_f, ), dtype=np.int8, buffer=yr_idx_f_shm.buf)
    np.copyto(shared_yr_idx_f, yr_idx_f_np)

    len_val_m = len(row_m)
    row_m_np = row_m.numpy().astype(np.int16)
    row_m_shm = shared_memory.SharedMemory(create=True, size=row_m_np.nbytes)
    shared_row_m = np.ndarray(
        (len_val_m, ), dtype=np.int16, buffer=row_m_shm.buf)
    np.copyto(shared_row_m, row_m_np)    
    col_m_np = col_m.numpy().astype(np.int16)
    col_m_shm = shared_memory.SharedMemory(create=True, size=col_m_np.nbytes)
    shared_col_m = np.ndarray(
        (len_val_m, ), dtype=np.int16, buffer=col_m_shm.buf)
    np.copyto(shared_col_m, col_m_np)
    q_val_m_np = q_val_m.numpy().astype(np.int8)
    q_val_m_shm = shared_memory.SharedMemory(
        create=True, size=q_val_m_np.nbytes)
    shared_q_val_m = np.ndarray(
        (len_val_m, ), dtype=np.int8, buffer=q_val_m_shm.buf)
    np.copyto(shared_q_val_m, q_val_m_np)
    p_val_m_np = p_val_m.numpy().astype(np.float32)
    p_val_m_shm = shared_memory.SharedMemory(
        create=True, size=p_val_m_np.nbytes)
    shared_p_val_m = np.ndarray(
        (len_val_m, ), dtype=np.float32, buffer=p_val_m_shm.buf)
    np.copyto(shared_p_val_m, p_val_m_np)
    tp_idx_m_np = tp_idx_m.numpy().astype(np.int8)
    tp_idx_m_shm = shared_memory.SharedMemory(
        create=True, size=tp_idx_m_np.nbytes)
    shared_tp_idx_m = np.ndarray(
        (nobs_m, ), dtype=np.int8, buffer=tp_idx_m_shm.buf)
    np.copyto(shared_tp_idx_m, tp_idx_m_np)
    delta_idx_m_np = delta_idx_m.numpy().astype(np.int16)
    delta_idx_m_shm = shared_memory.SharedMemory(
        create=True, size=delta_idx_m_np.nbytes)
    shared_delta_idx_m = np.ndarray(
        (nobs_m, ), dtype=np.int16, buffer=delta_idx_m_shm.buf)
    np.copyto(shared_delta_idx_m, delta_idx_m_np)
    log_w_m_np = log_w_m.numpy().astype(np.float32)
    log_w_m_shm = shared_memory.SharedMemory(
        create=True, size=log_w_m_np.nbytes)
    shared_log_w_m = np.ndarray(
        (nobs_m, ), dtype=np.float32, buffer=log_w_m_shm.buf)
    np.copyto(shared_log_w_m, log_w_m_np)
    trend_m_np = trend_m.numpy().astype(np.float32)
    trend_m_shm = shared_memory.SharedMemory(
        create=True, size=trend_m_np.nbytes)
    shared_trend_m = np.ndarray(
        (nobs_m, ), dtype=np.float32, buffer=trend_m_shm.buf)
    np.copyto(shared_trend_m, trend_m_np)
    yr_idx_m_np = yr_idx_m.astype(np.int8)
    yr_idx_m_shm = shared_memory.SharedMemory(
        create=True, size=yr_idx_m_np.nbytes)
    shared_yr_idx_m = np.ndarray(
        (nobs_m, ), dtype=np.int8, buffer=yr_idx_m_shm.buf)
    np.copyto(shared_yr_idx_m, yr_idx_m_np)

    len_val_c = len(row_c)
    row_c_np = row_c.numpy().astype(np.int16)
    row_c_shm = shared_memory.SharedMemory(create=True, size=row_c_np.nbytes)
    shared_row_c = np.ndarray(
        (len_val_c, ), dtype=np.int16, buffer=row_c_shm.buf)
    np.copyto(shared_row_c, row_c_np)    
    col_c_np = col_c.numpy().astype(np.int16)
    col_c_shm = shared_memory.SharedMemory(create=True, size=col_c_np.nbytes)
    shared_col_c = np.ndarray(
        (len_val_c, ), dtype=np.int16, buffer=col_c_shm.buf)
    np.copyto(shared_col_c, col_c_np)
    q_val_c_np = q_val_c.numpy().astype(np.int8)
    q_val_c_shm = shared_memory.SharedMemory(
        create=True, size=q_val_c_np.nbytes)
    shared_q_val_c = np.ndarray(
        (len_val_c, ), dtype=np.int8, buffer=q_val_c_shm.buf)
    np.copyto(shared_q_val_c, q_val_c_np)
    p_val_c_np = p_val_c.numpy().astype(np.float32)
    p_val_c_shm = shared_memory.SharedMemory(
        create=True, size=p_val_c_np.nbytes)
    shared_p_val_c = np.ndarray(
        (len_val_c, ), dtype=np.float32, buffer=p_val_c_shm.buf)
    np.copyto(shared_p_val_c, p_val_c_np)
    tp_idx_c_np = tp_idx_c.numpy().astype(np.int8)
    tp_idx_c_shm = shared_memory.SharedMemory(
        create=True, size=tp_idx_c_np.nbytes)
    shared_tp_idx_c = np.ndarray(
        (nobs_c, ), dtype=np.int8, buffer=tp_idx_c_shm.buf)
    np.copyto(shared_tp_idx_c, tp_idx_c_np)
    delta_idx_c_np = delta_idx_c.numpy().astype(np.int16)
    delta_idx_c_shm = shared_memory.SharedMemory(
        create=True, size=delta_idx_c_np.nbytes)
    shared_delta_idx_c = np.ndarray(
        (nobs_c, ), dtype=np.int16, buffer=delta_idx_c_shm.buf)
    np.copyto(shared_delta_idx_c, delta_idx_c_np)
    log_w_c_np = log_w_c.numpy().astype(np.float32)
    log_w_c_shm = shared_memory.SharedMemory(
        create=True, size=log_w_c_np.nbytes)
    shared_log_w_c = np.ndarray(
        (nobs_c, ), dtype=np.float32, buffer=log_w_c_shm.buf)
    np.copyto(shared_log_w_c, log_w_c_np)
    trend_c_np = trend_c.numpy().astype(np.float32)
    trend_c_shm = shared_memory.SharedMemory(
        create=True, size=trend_c_np.nbytes)
    shared_trend_c = np.ndarray(
        (nobs_c, ), dtype=np.float32, buffer=trend_c_shm.buf)
    np.copyto(shared_trend_c, trend_c_np)
    yr_idx_c_np = yr_idx_c.astype(np.int8)
    yr_idx_c_shm = shared_memory.SharedMemory(
        create=True, size=yr_idx_c_np.nbytes)
    shared_yr_idx_c = np.ndarray(
        (nobs_c, ), dtype=np.int8, buffer=yr_idx_c_shm.buf)
    np.copyto(shared_yr_idx_c, yr_idx_c_np)

    len_valdn_f = len(idx_valdn_f)
    len_valdn_val_f = len(q_valdn_f.coalesce().values())
    q_valdn_coo_f_np = q_valdn_f.coalesce().indices().numpy().astype(np.int16)
    q_valdn_coo_f_shm = shared_memory.SharedMemory(
        create=True, size=q_valdn_coo_f_np.nbytes)
    shared_q_valdn_coo_f = np.ndarray(
        (3, len_valdn_val_f), 
        dtype=np.int16, buffer=q_valdn_coo_f_shm.buf)
    np.copyto(shared_q_valdn_coo_f, q_valdn_coo_f_np)
    q_valdn_val_f_np = q_valdn_f.coalesce().values().numpy().astype(np.int16)
    q_valdn_val_f_shm = shared_memory.SharedMemory(
        create=True, size=q_valdn_val_f_np.nbytes)
    shared_q_valdn_val_f = np.ndarray(
        (len_valdn_val_f, ), 
        dtype=np.int16, buffer=q_valdn_val_f_shm.buf)
    np.copyto(shared_q_valdn_val_f, q_valdn_val_f_np)
    pq_valdn_f_np = pq_valdn_f.numpy().astype(np.float32)
    pq_valdn_f_shm = shared_memory.SharedMemory(
        create=True, size=pq_valdn_f_np.nbytes)
    shared_pq_valdn_f = np.ndarray(
        (len_valdn_f, n_imposters+1), 
        dtype=np.float32, buffer=pq_valdn_f_shm.buf)
    np.copyto(shared_pq_valdn_f, pq_valdn_f_np)
    tp_valdn_f_np = tp_valdn_f.numpy().astype(np.int8)
    tp_valdn_f_shm = shared_memory.SharedMemory(
        create=True, size=tp_valdn_f_np.nbytes)
    shared_tp_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.int8, buffer=tp_valdn_f_shm.buf)
    np.copyto(shared_tp_valdn_f, tp_valdn_f_np)
    log_w_valdn_f_np = log_w_valdn_f.numpy().astype(np.float32)
    log_w_valdn_f_shm = shared_memory.SharedMemory(
        create=True, size=log_w_valdn_f_np.nbytes)
    shared_log_w_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.float32, buffer=log_w_valdn_f_shm.buf)
    np.copyto(shared_log_w_valdn_f, log_w_valdn_f_np)
    trend_valdn_f_np = trend_valdn_f.numpy().astype(np.float32)
    trend_valdn_f_shm = shared_memory.SharedMemory(
        create=True, size=trend_valdn_f_np.nbytes)
    shared_trend_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.float32, buffer=trend_valdn_f_shm.buf)
    np.copyto(shared_trend_valdn_f, trend_valdn_f_np)
    delta_valdn_f_np = delta_valdn_f.numpy().astype(np.int16)
    delta_valdn_f_shm = shared_memory.SharedMemory(
        create=True, size=delta_valdn_f_np.nbytes)
    shared_delta_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.int16, buffer=delta_valdn_f_shm.buf)
    np.copyto(shared_delta_valdn_f, delta_valdn_f_np)
    yr_valdn_f_np = yr_valdn_f.astype(np.int8)
    yr_valdn_f_shm = shared_memory.SharedMemory(
        create=True, size=yr_valdn_f_np.nbytes)
    shared_yr_valdn_f = np.ndarray(
        (len_valdn_f, ), dtype=np.int8, buffer=yr_valdn_f_shm.buf)
    np.copyto(shared_yr_valdn_f, yr_valdn_f_np)

    len_valdn_m = len(idx_valdn_m)
    len_valdn_val_m = len(q_valdn_m.coalesce().values())
    q_valdn_coo_m_np = q_valdn_m.coalesce().indices().numpy().astype(np.int16)
    q_valdn_coo_m_shm = shared_memory.SharedMemory(
        create=True, size=q_valdn_coo_m_np.nbytes)
    shared_q_valdn_coo_m = np.ndarray(
        (3, len_valdn_val_m), 
        dtype=np.int16, buffer=q_valdn_coo_m_shm.buf)
    np.copyto(shared_q_valdn_coo_m, q_valdn_coo_m_np)
    q_valdn_val_m_np = q_valdn_m.coalesce().values().numpy().astype(np.int16)
    q_valdn_val_m_shm = shared_memory.SharedMemory(
        create=True, size=q_valdn_val_m_np.nbytes)
    shared_q_valdn_val_m = np.ndarray(
        (len_valdn_val_m, ), 
        dtype=np.int16, buffer=q_valdn_val_m_shm.buf)
    np.copyto(shared_q_valdn_val_m, q_valdn_val_m_np)
    pq_valdn_m_np = pq_valdn_m.numpy().astype(np.float32)
    pq_valdn_m_shm = shared_memory.SharedMemory(
        create=True, size=pq_valdn_m_np.nbytes)
    shared_pq_valdn_m = np.ndarray(
        (len_valdn_m, n_imposters+1), 
        dtype=np.float32, buffer=pq_valdn_m_shm.buf)
    np.copyto(shared_pq_valdn_m, pq_valdn_m_np)
    tp_valdn_m_np = tp_valdn_m.numpy().astype(np.int8)
    tp_valdn_m_shm = shared_memory.SharedMemory(
        create=True, size=tp_valdn_m_np.nbytes)
    shared_tp_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.int8, buffer=tp_valdn_m_shm.buf)
    np.copyto(shared_tp_valdn_m, tp_valdn_m_np)
    log_w_valdn_m_np = log_w_valdn_m.numpy().astype(np.float32)
    log_w_valdn_m_shm = shared_memory.SharedMemory(
        create=True, size=log_w_valdn_m_np.nbytes)
    shared_log_w_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.float32, buffer=log_w_valdn_m_shm.buf)
    np.copyto(shared_log_w_valdn_m, log_w_valdn_m_np)
    trend_valdn_m_np = trend_valdn_m.numpy().astype(np.float32)
    trend_valdn_m_shm = shared_memory.SharedMemory(
        create=True, size=trend_valdn_m_np.nbytes)
    shared_trend_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.float32, buffer=trend_valdn_m_shm.buf)
    np.copyto(shared_trend_valdn_m, trend_valdn_m_np)
    delta_valdn_m_np = delta_valdn_m.numpy().astype(np.int16)
    delta_valdn_m_shm = shared_memory.SharedMemory(
        create=True, size=delta_valdn_m_np.nbytes)
    shared_delta_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.int16, buffer=delta_valdn_m_shm.buf)
    np.copyto(shared_delta_valdn_m, delta_valdn_m_np)
    yr_valdn_m_np = yr_valdn_m.astype(np.int8)
    yr_valdn_m_shm = shared_memory.SharedMemory(
        create=True, size=yr_valdn_m_np.nbytes)
    shared_yr_valdn_m = np.ndarray(
        (len_valdn_m, ), dtype=np.int8, buffer=yr_valdn_m_shm.buf)
    np.copyto(shared_yr_valdn_m, yr_valdn_m_np)

    len_valdn_c = len(idx_valdn_c)
    len_valdn_val_c = len(q_valdn_c.coalesce().values())
    q_valdn_coo_c_np = q_valdn_c.coalesce().indices().numpy().astype(np.int16)
    q_valdn_coo_c_shm = shared_memory.SharedMemory(
        create=True, size=q_valdn_coo_c_np.nbytes)
    shared_q_valdn_coo_c = np.ndarray(
        (3, len_valdn_val_c), 
        dtype=np.int16, buffer=q_valdn_coo_c_shm.buf)
    np.copyto(shared_q_valdn_coo_c, q_valdn_coo_c_np)
    q_valdn_val_c_np = q_valdn_c.coalesce().values().numpy().astype(np.int16)
    q_valdn_val_c_shm = shared_memory.SharedMemory(
        create=True, size=q_valdn_val_c_np.nbytes)
    shared_q_valdn_val_c = np.ndarray(
        (len_valdn_val_c, ), 
        dtype=np.int16, buffer=q_valdn_val_c_shm.buf)
    np.copyto(shared_q_valdn_val_c, q_valdn_val_c_np)
    pq_valdn_c_np = pq_valdn_c.numpy().astype(np.float32)
    pq_valdn_c_shm = shared_memory.SharedMemory(
        create=True, size=pq_valdn_c_np.nbytes)
    shared_pq_valdn_c = np.ndarray(
        (len_valdn_c, n_imposters+1), 
        dtype=np.float32, buffer=pq_valdn_c_shm.buf)
    np.copyto(shared_pq_valdn_c, pq_valdn_c_np)
    tp_valdn_c_np = tp_valdn_c.numpy().astype(np.int8)
    tp_valdn_c_shm = shared_memory.SharedMemory(
        create=True, size=tp_valdn_c_np.nbytes)
    shared_tp_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.int8, buffer=tp_valdn_c_shm.buf)
    np.copyto(shared_tp_valdn_c, tp_valdn_c_np)
    log_w_valdn_c_np = log_w_valdn_c.numpy().astype(np.float32)
    log_w_valdn_c_shm = shared_memory.SharedMemory(
        create=True, size=log_w_valdn_c_np.nbytes)
    shared_log_w_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.float32, buffer=log_w_valdn_c_shm.buf)
    np.copyto(shared_log_w_valdn_c, log_w_valdn_c_np)
    trend_valdn_c_np = trend_valdn_c.numpy().astype(np.float32)
    trend_valdn_c_shm = shared_memory.SharedMemory(
        create=True, size=trend_valdn_c_np.nbytes)
    shared_trend_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.float32, buffer=trend_valdn_c_shm.buf)
    np.copyto(shared_trend_valdn_c, trend_valdn_c_np)
    delta_valdn_c_np = delta_valdn_c.numpy().astype(np.int16)
    delta_valdn_c_shm = shared_memory.SharedMemory(
        create=True, size=delta_valdn_c_np.nbytes)
    shared_delta_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.int16, buffer=delta_valdn_c_shm.buf)
    np.copyto(shared_delta_valdn_c, delta_valdn_c_np)
    yr_valdn_c_np = yr_valdn_c.astype(np.int8)
    yr_valdn_c_shm = shared_memory.SharedMemory(
        create=True, size=yr_valdn_c_np.nbytes)
    shared_yr_valdn_c = np.ndarray(
        (len_valdn_c, ), dtype=np.int8, buffer=yr_valdn_c_shm.buf)
    np.copyto(shared_yr_valdn_c, yr_valdn_c_np)

    # Parallel Processing --------------------
    seeds = torch.randint(0, 2**32-1, (n_bt, ), device=device)

    args_all = tuple((
        idx_bt_all[i], init, param_idx, delta_idx_2004,
        price_f, price_m, price_c, 
        candidate_f, candidate_m, candidate_c,
        bt_idx_f, bt_idx_m, bt_idx_c, 
        nobs_f, nobs_m, nobs_c, 
        n_upc_f, n_upc_m, n_upc_c, 
        len_train_f, len_train_m, len_train_c,
        len_valdn_f, len_valdn_m, len_valdn_c,
        len_val_f, len_val_m, len_val_c,
        len_valdn_val_f, len_valdn_val_m, len_valdn_val_c,
        row_f_shm.name, row_m_shm.name, row_c_shm.name, 
        col_f_shm.name, col_m_shm.name, col_c_shm.name, 
        q_val_f_shm.name, q_val_m_shm.name, q_val_c_shm.name, 
        p_val_f_shm.name, p_val_m_shm.name, p_val_c_shm.name, 
        tp_idx_f_shm.name, tp_idx_m_shm.name, tp_idx_c_shm.name, 
        delta_idx_f_shm.name, delta_idx_m_shm.name, delta_idx_c_shm.name, 
        log_w_f_shm.name, log_w_m_shm.name, log_w_c_shm.name, 
        trend_f_shm.name, trend_m_shm.name, trend_c_shm.name, 
        yr_idx_f_shm.name, yr_idx_m_shm.name, yr_idx_c_shm.name, 
        q_valdn_coo_f_shm.name, q_valdn_coo_m_shm.name, q_valdn_coo_c_shm.name, 
        q_valdn_val_f_shm.name, q_valdn_val_m_shm.name, q_valdn_val_c_shm.name, 
        pq_valdn_f_shm.name, pq_valdn_m_shm.name, pq_valdn_c_shm.name, 
        tp_valdn_f_shm.name, tp_valdn_m_shm.name, tp_valdn_c_shm.name, 
        log_w_valdn_f_shm.name, log_w_valdn_m_shm.name, log_w_valdn_c_shm.name,
        trend_valdn_f_shm.name, trend_valdn_m_shm.name, trend_valdn_c_shm.name, 
        delta_valdn_f_shm.name, delta_valdn_m_shm.name, delta_valdn_c_shm.name, 
        yr_valdn_f_shm.name, yr_valdn_m_shm.name, yr_valdn_c_shm.name, 
        batch_size_f, batch_size_m, batch_size_c, seeds[i]
    ) for i in range(n_bt))

    bt_results = torch.zeros(
        (n_bt, len(init[:, range(param_idx[11], param_idx[16])].squeeze())), 
        dtype=torch.float32)

    with Pool() as pool:
        for i, result in enumerate(
            tqdm(pool.imap_unordered(bootstrap, args_all), total=n_bt)
        ):
            bt_results[i, :] = result

    # Clean up shared memory
    row_f_shm.close()
    row_m_shm.close()
    row_c_shm.close()
    col_f_shm.close()
    col_m_shm.close()
    col_c_shm.close()
    q_val_f_shm.close()
    q_val_m_shm.close()
    q_val_c_shm.close()
    p_val_f_shm.close()
    p_val_m_shm.close()
    p_val_c_shm.close()
    tp_idx_f_shm.close()
    tp_idx_m_shm.close()
    tp_idx_c_shm.close()
    delta_idx_f_shm.close()
    delta_idx_m_shm.close()
    delta_idx_c_shm.close()
    log_w_f_shm.close()
    log_w_m_shm.close()
    log_w_c_shm.close()
    trend_f_shm.close()
    trend_m_shm.close()
    trend_c_shm.close()
    yr_idx_f_shm.close()
    yr_idx_m_shm.close()
    yr_idx_c_shm.close()

    q_valdn_coo_f_shm.close()
    q_valdn_coo_m_shm.close()
    q_valdn_coo_c_shm.close()
    q_valdn_val_f_shm.close()
    q_valdn_val_m_shm.close()
    q_valdn_val_c_shm.close()
    pq_valdn_f_shm.close()
    pq_valdn_m_shm.close()
    pq_valdn_c_shm.close()
    tp_valdn_f_shm.close()
    tp_valdn_m_shm.close()
    tp_valdn_c_shm.close()
    log_w_valdn_f_shm.close()
    log_w_valdn_m_shm.close()
    log_w_valdn_c_shm.close()
    trend_valdn_f_shm.close()
    trend_valdn_m_shm.close()
    trend_valdn_c_shm.close()
    delta_valdn_f_shm.close()
    delta_valdn_m_shm.close()
    delta_valdn_c_shm.close()
    yr_valdn_f_shm.close()
    yr_valdn_m_shm.close()
    yr_valdn_c_shm.close()

    row_f_shm.unlink()
    row_m_shm.unlink()
    row_c_shm.unlink()
    col_f_shm.unlink()
    col_m_shm.unlink()
    col_c_shm.unlink()
    q_val_f_shm.unlink()
    q_val_m_shm.unlink()
    q_val_c_shm.unlink()
    p_val_f_shm.unlink()
    p_val_m_shm.unlink()
    p_val_c_shm.unlink()
    tp_idx_f_shm.unlink()
    tp_idx_m_shm.unlink()
    tp_idx_c_shm.unlink()
    delta_idx_f_shm.unlink()
    delta_idx_m_shm.unlink()
    delta_idx_c_shm.unlink()
    log_w_f_shm.unlink()
    log_w_m_shm.unlink()
    log_w_c_shm.unlink()
    trend_f_shm.unlink()
    trend_m_shm.unlink()
    trend_c_shm.unlink()
    yr_idx_f_shm.unlink()
    yr_idx_m_shm.unlink()
    yr_idx_c_shm.unlink()
  
    q_valdn_coo_f_shm.unlink()
    q_valdn_coo_m_shm.unlink()
    q_valdn_coo_c_shm.unlink()
    q_valdn_val_f_shm.unlink()
    q_valdn_val_m_shm.unlink()
    q_valdn_val_c_shm.unlink()
    pq_valdn_f_shm.unlink()
    pq_valdn_m_shm.unlink() 
    pq_valdn_c_shm.unlink()
    tp_valdn_f_shm.unlink()
    tp_valdn_m_shm.unlink()
    tp_valdn_c_shm.unlink()
    log_w_valdn_f_shm.unlink()
    log_w_valdn_m_shm.unlink()
    log_w_valdn_c_shm.unlink()
    trend_valdn_f_shm.unlink()
    trend_valdn_m_shm.unlink()
    trend_valdn_c_shm.unlink()
    delta_valdn_f_shm.unlink()
    delta_valdn_m_shm.unlink()
    delta_valdn_c_shm.unlink()
    yr_valdn_f_shm.unlink()
    yr_valdn_m_shm.unlink()
    yr_valdn_c_shm.unlink()

    # Calculate standard errors --------------------
    # Add the original sample to the bootstrap outcome
    orig_params = params[
        :, range(param_idx[11], param_idx[16])].detach()
    bt_results = torch.cat((bt_results, orig_params), dim=0)

    # Structural parameters
    param_idx_new = param_idx - param_idx[11]
    param_std = np.std(bt_results.numpy(), axis=0)
    delta = orig_params[:, range(param_idx_new[11], param_idx_new[12])]
    delta_std = param_std[range(param_idx_new[11], param_idx_new[12])]
    lambda_w = orig_params[:, range(param_idx_new[13], param_idx_new[14])]
    lambda_w_std = param_std[range(param_idx_new[13], param_idx_new[14])]
    lambda_yr = orig_params[:, range(param_idx_new[14], param_idx_new[15])]
    lambda_yr_std = param_std[range(param_idx_new[14], param_idx_new[15])]
    lambda_yr_w = orig_params[:, range(param_idx_new[15], param_idx_new[16])]
    lambda_yr_w_std = param_std[range(param_idx_new[15], param_idx_new[16])]

    # Tests of lambda2+lambda3*t=1
    trend = {}

    for i, yr in enumerate(range(2004, 2021)):
        trend[str(yr)] = i / (len(range(2004, 2021))-1)

    test_no_w = {}

    for yr in range(2004, 2021):
        trend_yr = trend[str(yr)]
        cp_f_bt = np.empty(n_bt+1)
        cp_m_bt = np.empty(n_bt+1)
        one_f_bt = np.empty(n_bt+1)
        one_m_bt = np.empty(n_bt+1)
        one_c_bt = np.empty(n_bt+1)
        two_f_bt = np.empty(n_bt+1)
        two_m_bt = np.empty(n_bt+1)
        two_c_bt = np.empty(n_bt+1)

        for i in range(n_bt+1):
            lambda_w_bt = bt_results[
                i, range(param_idx_new[13], param_idx_new[14])]
            lambda_yr_w_bt = bt_results[
                i, range(param_idx_new[15], param_idx_new[16])]
            lambda_sum = lambda_w_bt + lambda_yr_w_bt*trend_yr

            cp_f_bt[i] = lambda_sum[0]
            cp_m_bt[i] = lambda_sum[3]
            one_f_bt[i] = lambda_sum[1]
            one_m_bt[i] = lambda_sum[4]
            one_c_bt[i] = lambda_sum[6]
            two_f_bt[i] = lambda_sum[2]
            two_m_bt[i] = lambda_sum[5]
            two_c_bt[i] = lambda_sum[7]

        test_no_w['0_'+str(yr)+'_0'] = {
            'est': cp_f_bt[n_bt], 
            'std': np.std(cp_f_bt)
        }
        test_no_w['0_'+str(yr)+'_1'] = {
            'est': one_f_bt[n_bt], 
            'std': np.std(one_f_bt)
        }
        test_no_w['0_'+str(yr)+'_2'] = {
            'est': two_f_bt[n_bt], 
            'std': np.std(two_f_bt)
        }
        test_no_w['1_'+str(yr)+'_0'] = {
            'est': cp_m_bt[n_bt], 
            'std': np.std(cp_m_bt)
        }
        test_no_w['1_'+str(yr)+'_1'] = {
            'est': one_m_bt[n_bt], 
            'std': np.std(one_m_bt)
        }
        test_no_w['1_'+str(yr)+'_2'] = {
            'est': two_m_bt[n_bt], 
            'std': np.std(two_m_bt)
        }
        test_no_w['2_'+str(yr)+'_1'] = {
            'est': one_c_bt[n_bt], 
            'std': np.std(one_c_bt)
        }
        test_no_w['2_'+str(yr)+'_2'] = {
            'est': two_c_bt[n_bt], 
            'std': np.std(two_c_bt)
        }
    
    # Marginal effects, budget
    w = np.array([1875, 2292, 2708, 3125, 3542, 3958, 4583, 5417])
    w_demean = np.array([np.log(1875) - log_w_mean,
                         np.log(2292) - log_w_mean,
                         np.log(2708) - log_w_mean,
                         np.log(3125) - log_w_mean,
                         np.log(3542) - log_w_mean,
                         np.log(3958) - log_w_mean,
                         np.log(4583) - log_w_mean,
                         np.log(5417) - log_w_mean])
    avg_marginal_w_bt = {}

    for i in range(n_bt+1):
        delta_bt = bt_results[i, range(param_idx_new[11], param_idx_new[12])]
        lambda_w_bt = bt_results[
            i, range(param_idx_new[13], param_idx_new[14])]
        lambda_yr_bt = bt_results[
            i, range(param_idx_new[14], param_idx_new[15])]
        lambda_yr_w_bt = bt_results[
            i, range(param_idx_new[15], param_idx_new[16])]
        beta2_cp_bt = np.exp(bt_results[i, param_idx_new[12]].item())
        beta2_one_bt = np.exp(bt_results[i, param_idx_new[12]+1].item())
        beta2_two_bt = np.exp(bt_results[i, param_idx_new[12]+2].item())
        avg_mar_w = {}
        
        for j in range(len(w)):
            w_marginal = w[j]
            w_demean_marginal = w_demean[j]
            mar_w = marginal_w(
                delta_bt, lambda_w_bt, lambda_yr_bt, lambda_yr_w_bt, 
                beta2_cp_bt, beta2_one_bt, beta2_two_bt, w_marginal, 
                w_demean_marginal)

            mar_w_cp_f = np.empty(17)
            mar_w_cp_m = np.empty(17)
            mar_w_one_f = np.empty(17)
            mar_w_one_m = np.empty(17)
            mar_w_one_c = np.empty(17)
            mar_w_two_f = np.empty(16)
            mar_w_two_m = np.empty(16)
            mar_w_two_c = np.empty(16)

            for k, yr in enumerate(range(2004, 2021)):
                mar_w_cp_f[k] = mar_w[str(yr)+'_0']['female']
                mar_w_cp_m[k] = mar_w[str(yr)+'_0']['male']
                mar_w_one_f[k] = mar_w[str(yr)+'_1']['female']
                mar_w_one_m[k] = mar_w[str(yr)+'_1']['male']
                mar_w_one_c[k] = mar_w[str(yr)+'_1']['children']
                if yr != 2020:
                    mar_w_two_f[k] = mar_w[str(yr)+'_2']['female']
                    mar_w_two_m[k] = mar_w[str(yr)+'_2']['male']
                    mar_w_two_c[k] = mar_w[str(yr)+'_2']['children']

            avg_mar_w['w'+str(j)] = {
                'cp_f': np.mean(mar_w_cp_f),
                'cp_m': np.mean(mar_w_cp_m),
                'one_f': np.mean(mar_w_one_f),
                'one_m': np.mean(mar_w_one_m),
                'one_c': np.mean(mar_w_one_c),
                'two_f': np.mean(mar_w_two_f),
                'two_m': np.mean(mar_w_two_m),
                'two_c': np.mean(mar_w_two_c),
            }
        
        avg_marginal_w_bt[i] = avg_mar_w

    tp_gd_key = avg_marginal_w_bt[0]['w0'].keys()
    avg_mar_dict = {}

    for key in tp_gd_key:
        avg_mar = np.empty(len(w))
        avg_mar_std = np.empty(len(w))

        for i in range(len(w)):
            avg_mar_bt = np.zeros(n_bt+1)

            for bt in range(n_bt+1):
                avg_mar_bt[bt] = avg_marginal_w_bt[bt]['w'+str(i)][key]
            
            avg_mar[i] = avg_mar_bt[n_bt]
            avg_mar_std[i] = np.std(avg_mar_bt)

        avg_mar_dict[key] = {'est': avg_mar, 'std': avg_mar_std}

    # Resource shares
    shares_bt = {}
    avg_shares_tp_bt = {}
    avg_shares_tp_w_bt = {}

    for i in range(n_bt+1):
        delta_bt = bt_results[i, range(param_idx_new[11], param_idx_new[12])]
        lambda_w_bt = bt_results[
            i, range(param_idx_new[13], param_idx_new[14])]
        lambda_yr_bt = bt_results[
            i, range(param_idx_new[14], param_idx_new[15])]
        lambda_yr_w_bt = bt_results[
            i, range(param_idx_new[15], param_idx_new[16])]
        beta2_cp_bt = np.exp(bt_results[i, param_idx_new[12]].item())
        beta2_one_bt = np.exp(bt_results[i, param_idx_new[12]+1].item())
        beta2_two_bt = np.exp(bt_results[i, param_idx_new[12]+2].item())

        (
            shares, avg_shares_tp, avg_shares_tp_w, unreasonable_shares
        ) = resource_share(delta_bt, lambda_w_bt, lambda_yr_bt, 
                           lambda_yr_w_bt, beta2_cp_bt, beta2_one_bt, 
                           beta2_two_bt)
        
        shares_bt[i] = shares
        avg_shares_tp_bt[i] = avg_shares_tp
        avg_shares_tp_w_bt[i] = avg_shares_tp_w

    avg_shares_tp_std = {}

    for tp in range(3):
        if tp == 0:
            f_bt = np.zeros(n_bt+1)
            m_bt = np.zeros(n_bt+1)

            for bt in range(n_bt+1):
                f_bt[bt] = avg_shares_tp_bt[bt][str(tp)]['female']
                m_bt[bt] = avg_shares_tp_bt[bt][str(tp)]['male']

            avg_shares_tp_std[str(tp)] = {'female': np.std(f_bt),
                                          'male': np.std(m_bt)}
        else:
            f_bt = np.zeros(n_bt+1)
            m_bt = np.zeros(n_bt+1)
            c_bt = np.zeros(n_bt+1)

            for bt in range(n_bt+1):
                f_bt[bt] = avg_shares_tp_bt[bt][str(tp)]['female']
                m_bt[bt] = avg_shares_tp_bt[bt][str(tp)]['male']
                c_bt[bt] = avg_shares_tp_bt[bt][str(tp)]['children']

            avg_shares_tp_std[str(tp)] = {'female': np.std(f_bt),
                                          'male': np.std(m_bt),
                                          'children': np.std(c_bt)}
            
    avg_shares_tp_w_std = {}

    for tp in avg_shares_tp_w.keys():
        if tp.split('_')[0] == '0':
            f_bt = np.zeros(n_bt+1)
            m_bt = np.zeros(n_bt+1)

            for bt in range(n_bt+1):
                f_bt[bt] = avg_shares_tp_w_bt[bt][str(tp)]['female']
                m_bt[bt] = avg_shares_tp_w_bt[bt][str(tp)]['male']

            avg_shares_tp_w_std[str(tp)] = {'female': np.std(f_bt),
                                            'male': np.std(m_bt)}
        else:
            f_bt = np.zeros(n_bt+1)
            m_bt = np.zeros(n_bt+1)
            c_bt = np.zeros(n_bt+1)

            for bt in range(n_bt+1):
                f_bt[bt] = avg_shares_tp_w_bt[bt][str(tp)]['female']
                m_bt[bt] = avg_shares_tp_w_bt[bt][str(tp)]['male']
                c_bt[bt] = avg_shares_tp_w_bt[bt][str(tp)]['children']

            avg_shares_tp_w_std[str(tp)] = {'female': np.std(f_bt),
                                            'male': np.std(m_bt),
                                            'children': np.std(c_bt)}
            
    # Gender gap
    avg_gap_bt = {}

    for i in range(n_bt+1):
        gap_cp = np.empty((17, 8))
        gap_one = np.empty((17, 8))
        gap_two = np.empty((16, 8))

        for yr in range(2004, 2021):
            gap_cp[yr-2004, :] = (
                shares_bt[i][str(yr)+'_0']['male'] 
                - shares_bt[i][str(yr)+'_0']['female']
            )
            gap_one[yr-2004, :] = (
                shares_bt[i][str(yr)+'_1']['male'] 
                - shares_bt[i][str(yr)+'_1']['female']
            )
            if yr != 2020:
                gap_two[yr-2004, :] = (
                    shares_bt[i][str(yr)+'_2']['male'] 
                    - shares_bt[i][str(yr)+'_2']['female']
                )

        avg_gap_bt[i] = {'cp': np.mean(gap_cp, axis=0),
                         'one': np.mean(gap_one, axis=0),
                         'two': np.mean(gap_two, axis=0)}
        
    avg_gap_dict = {}

    for tp in ['cp', 'one', 'two']:
        gap_bt = np.empty((n_bt+1, 8))

        for i in range(n_bt+1):
            gap_bt[i, :] = avg_gap_bt[i][tp]

        avg_gap_dict[tp] = {
            'est': gap_bt[n_bt, :],
            'std': np.std(gap_bt, axis=0)
        }
            
    # Beta 2
    beta2_bt = bt_results[:, param_idx_new[12]:(param_idx_new[12]+3)].numpy()
    beta2_std = np.std(beta2_bt, axis=0)

    # Tables --------------------
    # Structural Parameters
    columns = pd.MultiIndex.from_tuples(
        [(t, subcol) for t in ['Couples', 'One Child', 'Two Children'] 
         for subcol in ['female', 'male', 'children'] 
        if t != 'Couples' or subcol != 'children']
    )
    str_param_row = pd.MultiIndex.from_product([['$hat{lambda}_{0}$',
                                                 '$hat{lambda}_{2}$',
                                                 '$hat{lambda}_{3}$'], 
                                                [0, 1]])
    str_param_table = pd.DataFrame(index=str_param_row, 
                                    columns=columns)
    
    col_order = [0, 3, 1, 4, 6, 2, 5, 7]
    
    for i, col in enumerate(col_order):
        str_param_table.iloc[0, i] = '%.4f'%delta[:, i]
        str_param_table.iloc[1, i] = '(%.4f)'%delta_std[i]
        str_param_table.iloc[2, i] = '%.4f'%lambda_w[:, i]
        str_param_table.iloc[3, i] = '(%.4f)'%lambda_w_std[i]
        str_param_table.iloc[4, i] = '%.4f'%lambda_yr_w[:, i]
        str_param_table.iloc[5, i] = '(%.4f)'%lambda_yr_w_std[i]
    
    str_param_table.index = str_param_table.index.map(
        lambda x: (x[0], str(x[0]) if x[1] == 0 else ''))
    str_param_table = str_param_table.reset_index(level=0, drop=True)

    print(str_param_table.to_latex())

    # Resource shares
    types = ['0', '1', '2']
    income = ['0', '1', '2', '3', '4', '5', '6', '7']
    columns = pd.MultiIndex.from_tuples(
        [(t, subcol) for t in types for subcol in ['female', 'male', 'children'] 
        if t != '0' or subcol != 'children']
    )
    shares_table_std = pd.DataFrame(index=['Overall']+income, 
                                    columns=columns, 
                                    dtype=float)

    for t, results in avg_shares_tp_std.items():
        for subcol, value in results.items():
            shares_table_std.loc['Overall', (t, subcol)] = value

    for key, results in avg_shares_tp_w_std.items():
        t_type, income = key.split('_')[:2]
        for subcol, value in results.items():
            shares_table_std.loc[income, (t_type, subcol)] = value

    col_names = {'0': 'Couples', '1': 'One Child', '2': 'Two Children'}
    shares_table_std.columns = shares_table_std.columns.set_levels(
        [col_names.get(t, t) for t in shares_table_std.columns.levels[0]],
        level=0
    )
    row_names = {'0': '$22,500', 
                 '1': '$27,500', 
                 '2': '$32,500',
                 '3': '$37,500', 
                 '4': '$42,500', 
                 '5': '$47,500', 
                 '6': '$55,000', 
                 '7': '$65,000'}
    shares_table_std.rename(index=lambda x: row_names.get(x, x), inplace=True)

    shares_rows = pd.MultiIndex.from_product([shares_table.index, [0, 1]])
    shares_table_combined = pd.DataFrame(index=shares_rows, columns=columns)
    shares_table_combined.columns = shares_table_combined.columns.set_levels(
        [col_names.get(t, t) for t in shares_table_combined.columns.levels[0]],
        level=0
    )

    for row in shares_table.index:
        for col in shares_table_combined.columns:
            shares_table_combined.loc[(row, 0), col] = '%.4f'%(
                shares_table.loc[row, col])
            shares_table_combined.loc[(row, 1), col] = '(%.4f)'%(
                shares_table_std.loc[row, col])

    shares_table_combined.index = shares_table_combined.index.map(
        lambda x: (x[0], str(x[0]) if x[1] == 0 else ''))
    shares_table_combined = shares_table_combined.reset_index(
        level=0, drop=True)

    print(shares_table_combined.to_latex())

    # Tests of lambda2+lambda3*t=1
    year_list = [str(i) for i in range(2004, 2021)]
    est_sd_idx = pd.MultiIndex.from_product([year_list, [0, 1]])
    test_no_w_table = pd.DataFrame(index=est_sd_idx, columns=columns)
    
    for key in test_no_w.keys():
        gender, year, type_ = key.split("_")
        if gender == '0':
            gender = 'female'
        elif gender == '1':
            gender = 'male'
        else:
            gender = 'children'
        t_stat = (test_no_w[key]['est']-1) / test_no_w[key]['std']
        p_val = scipy.stats.norm.sf(abs(t_stat))*2

        test_no_w_table.loc[(year, 0), (type_, gender)] = '%.4f'%(
            test_no_w[key]['est'])
        if (p_val<=0.05) & (p_val>0.01):
            test_no_w_table.loc[(year, 1), (type_, gender)] = '(%.4f)$^{*}$'%(
                test_no_w[key]['std'])
        elif (p_val<=0.01) & (p_val>0.001):
            test_no_w_table.loc[(year, 1), (type_, gender)] = '(%.4f)$^{**}$'%(
                test_no_w[key]['std'])
        elif (p_val<=0.001):
            test_no_w_table.loc[
                (year, 1), 
                (type_, gender)
            ] = '(%.4f)$^{***}$'%(test_no_w[key]['std'])
        else:
            test_no_w_table.loc[(year, 1), (type_, gender)] = '(%.4f)'%(
                test_no_w[key]['std'])

    test_no_w_table.index = test_no_w_table.index.map(
        lambda x: (x[0], str(x[0]) if x[1] == 0 else ''))
    test_no_w_table = test_no_w_table.reset_index(level=0, drop=True)
    test_no_w_table.columns = test_no_w_table.columns.set_levels(
        [col_names.get(t, t) for t in test_no_w_table.columns.levels[0]],
        level=0
    )

    print(test_no_w_table.to_latex())

    # Gender gap
    budget_list = [row_names[str(i)] for i in range(len(w))]
    est_sd_idx = pd.MultiIndex.from_product([budget_list, [0, 1]])
    gap_table = pd.DataFrame(
        index=est_sd_idx, columns=['Couples', 'One Child', 'Two Children']
    )
    
    for i, key in enumerate(avg_gap_dict.keys()):
        t_stat = avg_gap_dict[key]['est'] / avg_gap_dict[key]['std']
        p_val = scipy.stats.norm.sf(abs(t_stat))*2
        col_val = []

        for j in range(len(w)):
            col_val.append('%.4f'%(avg_gap_dict[key]['est'][j]))
            if (p_val[j]<=0.05) & (p_val[j]>0.01):
                col_val.append('(%.4f)$^{*}$'%(avg_gap_dict[key]['std'][j]))
            elif (p_val[j]<=0.01) & (p_val[j]>0.001):
                col_val.append(
                    '(%.4f)$^{**}$'%(avg_gap_dict[key]['std'][j]))
            elif (p_val[j]<=0.001):
                col_val.append(
                    '(%.4f)$^{***}$'%(avg_gap_dict[key]['std'][j]))
            else:
                col_val.append('(%.4f)'%(avg_gap_dict[key]['std'][j]))
        
        gap_table.iloc[:, i] = col_val

    gap_table.index = gap_table.index.map(
        lambda x: (x[0], str(x[0]) if x[1] == 0 else ''))
    gap_table = gap_table.reset_index(level=0, drop=True)

    print(gap_table.to_latex())

    # Marginal effects, budget
    est_sd_idx = pd.MultiIndex.from_product([budget_list, [0, 1]])
    marginal_w_table = pd.DataFrame(index=est_sd_idx, columns=columns)
    
    for i, key in enumerate(avg_mar_dict.keys()):
        t_stat = avg_mar_dict[key]['est'] / avg_mar_dict[key]['std']
        p_val = scipy.stats.norm.sf(abs(t_stat))*2
        col_val = []

        for j in range(len(w)):
            col_val.append('%.4f'%(avg_mar_dict[key]['est'][j]*1e3))
            if (p_val[j]<=0.05) & (p_val[j]>0.01):
                col_val.append('(%.4f)$^{*}$'%(avg_mar_dict[key]['std'][j]*1e3))
            elif (p_val[j]<=0.01) & (p_val[j]>0.001):
                col_val.append(
                    '(%.4f)$^{**}$'%(avg_mar_dict[key]['std'][j]*1e3))
            elif (p_val[j]<=0.001):
                col_val.append(
                    '(%.4f)$^{***}$'%(avg_mar_dict[key]['std'][j]*1e3))
            else:
                col_val.append('(%.4f)'%(avg_mar_dict[key]['std'][j]*1e3))
        
        marginal_w_table.iloc[:, i] = col_val

    marginal_w_table.index = marginal_w_table.index.map(
        lambda x: (x[0], str(x[0]) if x[1] == 0 else ''))
    marginal_w_table = marginal_w_table.reset_index(level=0, drop=True)
    marginal_w_table.columns = marginal_w_table.columns.set_levels(
        [col_names.get(t, t) for t in marginal_w_table.columns.levels[0]],
        level=0
    )

    print(marginal_w_table.to_latex())

    # Graphs --------------------
    # Recource shares, by year
    shares_cp_f = {
        yr: shares_bt[n_bt][str(yr)+'_0']['female'] for yr in range(2004, 2021)}
    shares_cp_m = {
        yr: shares_bt[n_bt][str(yr)+'_0']['male'] for yr in range(2004, 2021)}
    shares_one_f = {
        yr: shares_bt[n_bt][str(yr)+'_1']['female'] for yr in range(2004, 2021)}
    shares_one_m = {
        yr: shares_bt[n_bt][str(yr)+'_1']['male'] for yr in range(2004, 2021)}
    shares_one_c = {
        yr: shares_bt[n_bt][str(yr)+'_1']['children'] 
        for yr in range(2004, 2021)}
    shares_two_f = {
        yr: shares_bt[n_bt][str(yr)+'_2']['female'] for yr in range(2004, 2020)}
    shares_two_m = {
        yr: shares_bt[n_bt][str(yr)+'_2']['male'] for yr in range(2004, 2020)}
    shares_two_c = {
        yr: shares_bt[n_bt][str(yr)+'_2']['children'] 
        for yr in range(2004, 2020)}

    shares_std_cp_f = {}
    shares_std_cp_m = {}
    shares_std_one_f = {}
    shares_std_one_m = {}
    shares_std_one_c = {}

    for yr in range(2004, 2021):
        shares_std_yr_cp_f = np.empty((n_bt+1, 8))
        shares_std_yr_cp_m = np.empty((n_bt+1, 8))
        shares_std_yr_one_f = np.empty((n_bt+1, 8))
        shares_std_yr_one_m = np.empty((n_bt+1, 8))
        shares_std_yr_one_c = np.empty((n_bt+1, 8))

        for i in range(n_bt+1):
            shares_std_yr_cp_f[i, :] = shares_bt[i][str(yr)+'_0']['female']
            shares_std_yr_cp_m[i, :] = shares_bt[i][str(yr)+'_0']['male']
            shares_std_yr_one_f[i, :] = shares_bt[i][str(yr)+'_1']['female']
            shares_std_yr_one_m[i, :] = shares_bt[i][str(yr)+'_1']['male']
            shares_std_yr_one_c[i, :] = shares_bt[i][str(yr)+'_1']['children']

        shares_std_cp_f[yr] = np.std(shares_std_yr_cp_f, axis=0)
        shares_std_cp_m[yr] = np.std(shares_std_yr_cp_m, axis=0)
        shares_std_one_f[yr] = np.std(shares_std_yr_one_f, axis=0)
        shares_std_one_m[yr] = np.std(shares_std_yr_one_m, axis=0)
        shares_std_one_c[yr] = np.std(shares_std_yr_one_c, axis=0)

    shares_std_two_f = {}
    shares_std_two_m = {}
    shares_std_two_c = {}

    for yr in range(2004, 2020):
        shares_std_yr_two_f = np.empty((n_bt+1, 8))
        shares_std_yr_two_m = np.empty((n_bt+1, 8))
        shares_std_yr_two_c = np.empty((n_bt+1, 8))

        for i in range(n_bt+1):
            shares_std_yr_two_f[i, :] = shares_bt[i][str(yr)+'_2']['female']
            shares_std_yr_two_m[i, :] = shares_bt[i][str(yr)+'_2']['male']
            shares_std_yr_two_c[i, :] = shares_bt[i][str(yr)+'_2']['children']

        shares_std_two_f[yr] = np.std(shares_std_yr_two_f, axis=0)
        shares_std_two_m[yr] = np.std(shares_std_yr_two_m, axis=0)
        shares_std_two_c[yr] = np.std(shares_std_yr_two_c, axis=0)

    z = scipy.stats.norm.ppf(.975)
    _, axes_cp = plt.subplots(4, 2, figsize=(14, 16))

    for i, ax in enumerate(axes_cp.flatten()):
        budget_level = row_names[str(i)]
        female_line, = ax.plot(
            range(2004, 2021), 
            [shares_cp_f[yr][i].item() for yr in range(2004, 2021)], 
            'o', label='Female'
        )
        female_color = female_line.get_color()
        male_line, = ax.plot(
            range(2004, 2021), 
            [shares_cp_m[yr][i].item() for yr in range(2004, 2021)], 
            'o', label='Male'
        )
        male_color = male_line.get_color()

        for yr in range(2004, 2021):
            ax.plot([yr, yr], [
                shares_cp_f[yr][i] - z * shares_std_cp_f[yr][i], 
                shares_cp_f[yr][i] + z * shares_std_cp_f[yr][i]
            ], color=female_color, linewidth=3)
            ax.plot([yr, yr], [
                shares_cp_m[yr][i] - z * shares_std_cp_m[yr][i], 
                shares_cp_m[yr][i] + z * shares_std_cp_m[yr][i]
            ], color=male_color, linewidth=3)

        ax.set_title(f'Annual Budget = {budget_level}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Share')
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

    _, axes_one = plt.subplots(4, 2, figsize=(14, 16))

    for i, ax in enumerate(axes_one.flatten()):
        budget_level = row_names[str(i)]
        female_line, = ax.plot(
            range(2004, 2021), 
            [shares_one_f[yr][i].item() for yr in range(2004, 2021)], 
            'o', label='Female'
        )
        female_color = female_line.get_color()
        male_line, = ax.plot(
            range(2004, 2021), 
            [shares_one_m[yr][i].item() for yr in range(2004, 2021)], 
            'o', label='Male'
        )
        male_color = male_line.get_color()
        children_line, = ax.plot(
            range(2004, 2021), 
            [shares_one_c[yr][i].item() for yr in range(2004, 2021)], 
            'o', label='Children'
        )
        children_color = children_line.get_color()

        for yr in range(2004, 2021):
            ax.plot([yr, yr], [
                shares_one_f[yr][i] - z * shares_std_one_f[yr][i], 
                shares_one_f[yr][i] + z * shares_std_one_f[yr][i]
            ], color=female_color, linewidth=3)
            ax.plot([yr, yr], [
                shares_one_m[yr][i] - z * shares_std_one_m[yr][i], 
                shares_one_m[yr][i] + z * shares_std_one_m[yr][i]
            ], color=male_color, linewidth=3)
            ax.plot([yr, yr], [
                shares_one_c[yr][i] - z * shares_std_one_c[yr][i], 
                shares_one_c[yr][i] + z * shares_std_one_c[yr][i]
            ], color=children_color, linewidth=3)

        ax.set_title(f'Annual Budget = {budget_level}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Share')
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

    _, axes_two = plt.subplots(4, 2, figsize=(14, 16))

    for i, ax in enumerate(axes_two.flatten()):
        budget_level = row_names[str(i)]
        female_line, = ax.plot(
            range(2004, 2020), 
            [shares_two_f[yr][i].item() for yr in range(2004, 2020)], 
            'o', label='Female'
        )
        female_color = female_line.get_color()
        male_line, = ax.plot(
            range(2004, 2020), 
            [shares_two_m[yr][i].item() for yr in range(2004, 2020)], 
            'o', label='Male'
        )
        male_color = male_line.get_color()
        children_line, = ax.plot(
            range(2004, 2020), 
            [shares_two_c[yr][i].item() for yr in range(2004, 2020)], 
            'o', label='Children'
        )
        children_color = children_line.get_color()

        for yr in range(2004, 2020):
            ax.plot([yr, yr], [
                shares_two_f[yr][i] - z * shares_std_two_f[yr][i], 
                shares_two_f[yr][i] + z * shares_std_two_f[yr][i]
            ], color=female_color, linewidth=3)
            ax.plot([yr, yr], [
                shares_two_m[yr][i] - z * shares_std_two_m[yr][i], 
                shares_two_m[yr][i] + z * shares_std_two_m[yr][i]
            ], color=male_color, linewidth=3)
            ax.plot([yr, yr], [
                shares_two_c[yr][i] - z * shares_std_two_c[yr][i], 
                shares_two_c[yr][i] + z * shares_std_two_c[yr][i]
            ], color=children_color, linewidth=3)

        ax.set_title(f'Annual Budget = {budget_level}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Share')
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Recource shares, by budget
    _, axes_cp = plt.subplots(9, 2, figsize=(14, 38))

    for i, ax in enumerate(axes_cp.flatten()[range(17)]):
        year = 2004 + i
        ax.plot(budget_list, shares_cp_f[year], color='C0', label='Female')
        ax.fill_between(
            budget_list, 
            shares_cp_f[year]-z*shares_std_cp_f[year], 
            shares_cp_f[year]+z*shares_std_cp_f[year], 
            color='C0', alpha=.3)
        ax.plot(budget_list, shares_cp_m[year], color='C1', label='Male')
        ax.fill_between(
            budget_list, 
            shares_cp_m[year]-z*shares_std_cp_m[year], 
            shares_cp_m[year]+z*shares_std_cp_m[year], 
            color='C1', alpha=.3)
        ax.set_title(f'Year = {year}')
        ax.set_xlabel('Annual Budget')
        ax.set_ylabel('Share')
        ax.set_ylim(0.45, 0.55)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

    _, axes_one = plt.subplots(9, 2, figsize=(14, 38))

    for i, ax in enumerate(axes_one.flatten()[range(17)]):
        year = 2004 + i
        ax.plot(budget_list, shares_one_f[year], color='C0', label='Female')
        ax.fill_between(
            budget_list, 
            shares_one_f[year]-z*shares_std_one_f[year], 
            shares_one_f[year]+z*shares_std_one_f[year], 
            color='C0', alpha=.3)
        ax.plot(budget_list, shares_one_m[year], color='C1', label='Male')
        ax.fill_between(
            budget_list, 
            shares_one_m[year]-z*shares_std_one_m[year], 
            shares_one_m[year]+z*shares_std_one_m[year], 
            color='C1', alpha=.3)
        ax.plot(budget_list, shares_one_c[year], color='C2', label='Children')
        ax.fill_between(
            budget_list, 
            shares_one_c[year]-z*shares_std_one_c[year], 
            shares_one_c[year]+z*shares_std_one_c[year], 
            color='C2', alpha=.3)
        ax.set_title(f'Year = {year}')
        ax.set_xlabel('Annual Budget')
        ax.set_ylabel('Share')
        ax.set_ylim(0.2, 0.5)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

    _, axes_two = plt.subplots(9, 2, figsize=(14, 38))

    for i, ax in enumerate(axes_two.flatten()[range(16)]):
        year = 2004 + i
        ax.plot(budget_list, shares_two_f[year], color='C0', label='Female')
        ax.fill_between(
            budget_list, 
            shares_two_f[year]-z*shares_std_two_f[year], 
            shares_two_f[year]+z*shares_std_two_f[year], 
            color='C0', alpha=.3)
        ax.plot(budget_list, shares_two_m[year], color='C1', label='Male')
        ax.fill_between(
            budget_list, 
            shares_two_m[year]-z*shares_std_two_m[year], 
            shares_two_m[year]+z*shares_std_two_m[year], 
            color='C1', alpha=.3)
        ax.plot(budget_list, shares_two_c[year], color='C2', label='Children')
        ax.fill_between(
            budget_list, 
            shares_two_c[year]-z*shares_std_two_c[year], 
            shares_two_c[year]+z*shares_std_two_c[year], 
            color='C2', alpha=.3)
        ax.set_title(f'Year = {year}')
        ax.set_xlabel('Annual Budget')
        ax.set_ylabel('Share')
        ax.set_ylim(0.2, 0.5)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Marginal effects, budget
    _, axes = plt.subplots(4, 2, figsize=(14, 16))

    for i, ax in enumerate(axes.flatten()):
        key = list(tp_gd_key)[i]
        type_, gender = key.split("_")

        if gender == 'f':
            clr = 'C0'
            gender = 'Female'
        elif gender == 'm':
            clr = 'C1'
            gender = 'Male'
        else:
            clr = 'C2'
            gender = 'Children'

        if type_ == 'cp':
            type_ = 'Couples'
        elif type_ == 'one':
            type_ = 'One Child'
        else:
            type_ = 'Two Children'

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.plot(budget_list, avg_mar_dict[key]['est']*1e3, color=clr)
        ax.fill_between(
            budget_list, 
            (avg_mar_dict[key]['est']-z*avg_mar_dict[key]['std'])*1e3, 
            (avg_mar_dict[key]['est']+z*avg_mar_dict[key]['std'])*1e3, 
            color=clr, alpha=.3)
        ax.set_title(type_+', '+gender)
        ax.set_xlabel("Annual Budget")
        ax.set_ylabel("Average Marginal Effect")
        ax.set_ylim(-0.1, 0.1)
        ax.grid()

    plt.tight_layout()
    plt.show()