import torch
import math
import dill

import matplotlib.pyplot as plt

# Parameters -------------------
n_bundles = 2048
n_purchased = 3
q_max = 3
n_upc = 100
n_attr = 5
n_imposters = 50
batch_size = 1024
device = torch.device("cpu")
torch.manual_seed(123)

# DGP -------------------
true_gama_f = torch.empty((n_attr, n_upc), device=device).uniform_(-1, 2)
true_gama_m = torch.empty((n_attr, n_upc), device=device).uniform_(-1, 2)
true_alfa_f = torch.empty((n_attr, 1), device=device).normal_(2, 1)
true_alfa_m = torch.empty((n_attr, 1), device=device).normal_(2, 1)
true_beta1 = 2
true_beta2 = 0.01
w = 150
eta_f = 0.6
eta_m = 0.4
true_delta_f = true_beta1 - 2*eta_f*true_beta2*w
true_delta_m = true_beta1 - 2*eta_m*true_beta2*w

p_f = torch.empty((n_upc, 1), device=device).uniform_(0.5, 5)
p_m = torch.empty((n_upc, 1), device=device).uniform_(0.5, 5)

# Metropolis-Hastings
q_col_f = torch.randperm(n_upc, device=device)[:n_purchased] 
q_col_m = torch.randperm(n_upc, device=device)[:n_purchased] 

q_val_f = torch.randint(
    1, q_max+1, size=(n_purchased, ), device=device).to(torch.float32)
q_val_m = torch.randint(
    1, q_max+1, size=(n_purchased, ), device=device).to(torch.float32)

attr_f = true_gama_f[:, q_col_f] @ q_val_f
pq_f = q_val_f @ p_f[q_col_f]
v1_f = true_alfa_f.T@attr_f - attr_f@attr_f
v2_f = true_delta_f*pq_f + true_beta2*pq_f*pq_f
v_f = v1_f - v2_f

attr_m = true_gama_m[:, q_col_m] @ q_val_m
pq_m = q_val_m @ p_m[q_col_m]
v1_m = true_alfa_m.T@attr_m - attr_m@attr_m
v2_m = true_delta_m*pq_m + true_beta2*pq_m*pq_m
v_m = v1_m - v2_m

q_col_f_all = []
q_val_f_all = []
q_col_m_all = []
q_val_m_all = []
p_f_all = []
p_m_all = []
t = 0
obs = 0

while obs < n_bundles:
    idx_del_f = torch.randint(0, n_purchased, size=(1, 1), device=device).item()
    idx_del_m = torch.randint(0, n_purchased, size=(1, 1), device=device).item()

    qi_col_f = torch.arange(n_upc, device=device)
    qi_col_f = qi_col_f[~torch.isin(qi_col_f, q_col_f)]
    qi_col_f = qi_col_f[
        torch.randperm(n_upc-n_purchased, device=device)[0]].view(1, )
    qi_col_f = torch.cat((q_col_f[:idx_del_f], q_col_f[idx_del_f+1:], qi_col_f))

    qi_col_m = torch.arange(n_upc, device=device)
    qi_col_m = qi_col_m[~torch.isin(qi_col_m, q_col_m)]
    qi_col_m = qi_col_m[
        torch.randperm(n_upc-n_purchased, device=device)[0]].view(1, )
    qi_col_m = torch.cat((q_col_m[:idx_del_m], q_col_m[idx_del_m+1:], qi_col_m))

    qi_val_f = torch.randint(
        1, q_max+1, size=(1, ), device=device).to(torch.float32)
    qi_val_f = torch.cat((q_val_f[:idx_del_f], q_val_f[idx_del_f+1:], qi_val_f))

    qi_val_m = torch.randint(
        1, q_max+1, size=(1, ), device=device).to(torch.float32)
    qi_val_m = torch.cat((q_val_m[:idx_del_m], q_val_m[idx_del_m+1:], qi_val_m))

    p_f_new = p_f + torch.empty((n_upc, 1), device=device).normal_(0, 0.01)
    p_m_new = p_m + torch.empty((n_upc, 1), device=device).normal_(0, 0.01)

    attri_f = true_gama_f[:, qi_col_f] @ qi_val_f
    pqi_f = qi_val_f @ p_f_new[qi_col_f]
    v1i_f = true_alfa_f.T@attri_f - attri_f@attri_f
    v2i_f = true_delta_f*pqi_f + true_beta2*pqi_f*pqi_f
    vi_f = v1i_f - v2i_f

    attri_m = true_gama_m[:, qi_col_m] @ qi_val_m
    pqi_m = qi_val_m @ p_m_new[qi_col_m]
    v1i_m = true_alfa_m.T@attri_m - attri_m@attri_m
    v2i_m = true_delta_m*pqi_m + true_beta2*pqi_m*pqi_m
    vi_m = v1i_m - v2i_m

    u_f = torch.rand((1,), device=device)
    u_m = torch.rand((1,), device=device)

    if (torch.exp(vi_f - v_f) >= u_f).item():
        q_col_f = qi_col_f[:]
        q_val_f = qi_val_f[:]
        v_f = vi_f[:]

    if (torch.exp(vi_m - v_m) >= u_m).item():
        q_col_m = qi_col_m[:]
        q_val_m = qi_val_m[:]
        v_m = vi_m[:]

    if t >= 5000:
        if t % 200 == 0:
            q_col_f_all.append(q_col_f)
            q_val_f_all.append(q_val_f)
            q_col_m_all.append(q_col_m)
            q_val_m_all.append(q_val_m)
            p_f_all.append(p_f_new.squeeze_())
            p_m_all.append(p_m_new.squeeze_())
            obs += 1

    t += 1

q_col_f_all = torch.stack(q_col_f_all, dim=0)
q_val_f_all = torch.stack(q_val_f_all, dim=0)
q_col_m_all = torch.stack(q_col_m_all, dim=0)
q_val_m_all = torch.stack(q_val_m_all, dim=0)
p_f_all = torch.stack(p_f_all, dim=0)
p_m_all = torch.stack(p_m_all, dim=0)

# Exclude nonselected items
upc_selected_f = torch.unique(q_col_f_all)
upc_selected_m = torch.unique(q_col_m_all)
n_upc_f = len(upc_selected_f)
n_upc_m = len(upc_selected_m)

if n_upc_f < n_upc:
    idx_range = torch.arange(n_upc, device=device)
    missing = idx_range[~torch.isin(idx_range, upc_selected_f)]

    for idx in missing:
        missing[missing > idx] -= 1

    for idx in missing:
        q_col_f_all[q_col_f_all > idx] -= 1

if n_upc_m < n_upc:
    idx_range = torch.arange(n_upc, device=device)
    missing = idx_range[~torch.isin(idx_range, upc_selected_m)]

    for idx in missing:
        missing[missing > idx] -= 1

    for idx in missing:
        q_col_m_all[q_col_m_all > idx] -= 1

true_gama_f = true_gama_f[:, upc_selected_f]
true_gama_m = true_gama_m[:, upc_selected_m]
p_f_all = p_f_all[:, upc_selected_f]
p_m_all = p_m_all[:, upc_selected_m]

len_gama_f = n_attr*n_upc_f
len_gama_m = n_attr*n_upc_m
len_alfa = n_attr

param_idx = torch.cumsum(
    torch.tensor([0, len_gama_f, len_gama_m, len_alfa, len_alfa, 1, 1, 1], 
                 device=device), dim=0)

q_f = torch.zeros((n_bundles, n_upc_f), device=device)
q_m = torch.zeros((n_bundles, n_upc_m), device=device)
q_f = q_f.scatter_(1, q_col_f_all, q_val_f_all)
q_m = q_m.scatter_(1, q_col_m_all, q_val_m_all)

# generate imposters -------------------
idx_del_f = torch.randint(high=n_purchased, size=(n_bundles, ), device=device)
idx_del_m = torch.randint(high=n_purchased, size=(n_bundles, ), device=device)

mask_f = torch.ones((n_bundles, n_upc_f), dtype=torch.bool, device=device)
mask_m = torch.ones((n_bundles, n_upc_m), dtype=torch.bool, device=device)
mask_f[
    torch.arange(n_bundles, device=device).repeat_interleave(n_purchased), 
    q_col_f_all.flatten()] = False
mask_m[
    torch.arange(n_bundles, device=device).repeat_interleave(n_purchased), 
    q_col_m_all.flatten()] = False

candidate_f = torch.nonzero(mask_f, as_tuple=True)[1].reshape(n_bundles, -1)
candidate_m = torch.nonzero(mask_m, as_tuple=True)[1].reshape(n_bundles, -1)

selected_f_idx = torch.stack(
    [torch.randperm(n_upc_f-n_purchased, device=device)[:n_imposters] 
    for _ in range(n_bundles)])
selected_m_idx = torch.stack(
    [torch.randperm(n_upc_m-n_purchased, device=device)[:n_imposters] 
    for _ in range(n_bundles)])

selected_f = candidate_f.gather(1, selected_f_idx)
selected_m = candidate_m.gather(1, selected_m_idx)

qi_col_f = torch.cat((q_col_f_all[
    torch.arange(
        n_purchased, device=device
        ).repeat(n_bundles, 1) != idx_del_f.view(-1, 1)
    ].view(n_bundles, 
           n_purchased-1).unsqueeze_(-1).expand(n_bundles, 
                                                n_purchased-1, 
                                                n_imposters),
    selected_f.view(n_bundles, 1, n_imposters)), dim=1)
qi_col_m = torch.cat((q_col_m_all[
    torch.arange(
        n_purchased, device=device
        ).repeat(n_bundles, 1) != idx_del_m.view(-1, 1)
    ].view(n_bundles, 
           n_purchased-1).unsqueeze_(-1).expand(n_bundles, 
                                                n_purchased-1, 
                                                n_imposters),
    selected_m.view(n_bundles, 1, n_imposters)), dim=1)

qi_val_f_new = torch.randint(
    low=1, high=q_max+1, size=(n_bundles, 1, n_imposters), device=device
).float()
qi_val_m_new = torch.randint(
    low=1, high=q_max+1, size=(n_bundles, 1, n_imposters), device=device
).float()

qi_val_f = torch.cat((q_val_f_all[
    torch.arange(
        n_purchased, device=device
        ).repeat(n_bundles, 1) != idx_del_f.view(-1, 1)
    ].view(n_bundles, 
           n_purchased-1).unsqueeze_(-1).expand(n_bundles, 
                                                n_purchased-1, 
                                                n_imposters),
    qi_val_f_new), dim=1)
qi_val_m = torch.cat((q_val_m_all[
    torch.arange(
        n_purchased, device=device
        ).repeat(n_bundles, 1) != idx_del_m.view(-1, 1)
    ].view(n_bundles, 
           n_purchased-1).unsqueeze_(-1).expand(n_bundles, 
                                                n_purchased-1, 
                                                n_imposters),
    qi_val_m_new), dim=1)

q_f_all = [q_f]
q_m_all = [q_m]

for i in range(n_imposters):
    qi_f = torch.zeros((n_bundles, n_upc_f), device=device)
    qi_m = torch.zeros((n_bundles, n_upc_m), device=device)
    qi_f = qi_f.scatter_(1, qi_col_f[..., i], qi_val_f[..., i])
    qi_m = qi_m.scatter_(1, qi_col_m[..., i], qi_val_m[..., i])

    q_f_all.append(qi_f)
    q_m_all.append(qi_m)

q_f_all = torch.stack(q_f_all, dim=2)
q_m_all = torch.stack(q_m_all, dim=2)

# Maximum Likelihood -------------------
def mle(init, q_f_batch, q_m_batch, p_f_batch, p_m_batch):
    gama_f = init[:, range(param_idx[0], param_idx[1])].view(n_attr, n_upc_f)
    gama_m = init[:, range(param_idx[1], param_idx[2])].view(n_attr, n_upc_m)
    alfa_f = init[:, range(param_idx[2], param_idx[3])].view(n_attr, 1)
    alfa_m = init[:, range(param_idx[3], param_idx[4])].view(n_attr, 1)
    delta_f = torch.exp(init[:, param_idx[4]])
    delta_m = torch.exp(init[:, param_idx[5]])
    beta2 = torch.exp(init[:, param_idx[6]])
    n_obs_f = q_f_batch.shape[0]
    n_obs_m = q_m_batch.shape[0]

    attr_f = torch.matmul(q_f_batch.transpose(1, 2), gama_f.T)
    attr_m = torch.matmul(q_m_batch.transpose(1, 2), gama_m.T)

    pq_f = torch.matmul(
        q_f_batch.transpose(1, 2), p_f_batch.unsqueeze(-1)).squeeze_()
    pq_m = torch.matmul(
        q_m_batch.transpose(1, 2), p_m_batch.unsqueeze(-1)).squeeze_()

    vi_1_f = torch.matmul(attr_f, alfa_f).squeeze_()
    vi_1_m = torch.matmul(attr_m, alfa_m).squeeze_()

    vi_2_f = torch.matmul(
        attr_f.permute(1, 0, 2), attr_f.permute(1, 2, 0)
    ).diagonal(dim1=1, dim2=2).transpose(1, 0)
    vi_2_m = torch.matmul(
        attr_m.permute(1, 0, 2), attr_m.permute(1, 2, 0)
    ).diagonal(dim1=1, dim2=2).transpose(1, 0)

    vi_3_f = delta_f * pq_f
    vi_3_m = delta_m * pq_m

    vi_4_f = beta2 * pq_f * pq_f
    vi_4_m = beta2 * pq_m * pq_m

    vi_f = vi_1_f - vi_2_f - vi_3_f - vi_4_f
    vi_m = vi_1_m - vi_2_m - vi_3_m - vi_4_m

    # Numerically stable logit functions
    row_idx_f = torch.arange(n_obs_f, device=device)
    row_idx_m = torch.arange(n_obs_m, device=device)
    vi_max_f = torch.max(vi_f, axis=1)[0]
    vi_max_m = torch.max(vi_m, axis=1)[0]

    logit_f_1 = vi_f[row_idx_f, 0] - vi_max_f
    logit_m_1 = vi_m[row_idx_m, 0] - vi_max_m

    logit_f_2 = torch.sum(
        torch.exp(vi_f - vi_max_f.view(n_obs_f, 1)), dim=1)
    logit_m_2 = torch.sum(
        torch.exp(vi_m - vi_max_m.view(n_obs_m, 1)), dim=1)

    logit_f = torch.sum(logit_f_1 - torch.log(logit_f_2))
    logit_m = torch.sum(logit_m_1 - torch.log(logit_m_2))

    return -((logit_f+logit_m) / (n_obs_f+n_obs_m))

# Initialize parameters -------------------
true_param = torch.cat((
    true_gama_f.flatten(), 
    true_gama_m.flatten(),
    true_alfa_f.flatten(),
    true_alfa_m.flatten(),
    torch.tensor([true_delta_f], device=device),
    torch.tensor([true_delta_m], device=device),
    torch.tensor([true_beta2], device=device))).view(1, param_idx[-1])

init_gama_f = torch.empty((n_attr, n_upc_f), device=device).uniform_(-1, 2)
init_gama_m = torch.empty((n_attr, n_upc_m), device=device).uniform_(-1, 2)
init_alfa_f = torch.empty((n_attr, 1), device=device).normal_(2, 1)
init_alfa_m = torch.empty((n_attr, 1), device=device).normal_(2, 1)
init_delta_f = 0.4
init_delta_m = 0.5
init_beta2 = 0.1
init = torch.cat((
    init_gama_f.flatten(), 
    init_gama_m.flatten(),
    init_alfa_f.flatten(),
    init_alfa_m.flatten(),
    torch.log(torch.tensor([init_delta_f], device=device)),
    torch.log(torch.tensor([init_delta_m], device=device)),
    torch.log(torch.tensor([init_beta2], device=device)))).view(
        1, param_idx[-1]).requires_grad_()

# Gradient descent -------------------
# Hyperparameters
lr = 5e-3
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
tol = 1e-5
max_epoch = 10000

# Store the results
m = torch.zeros_like(init, device=device)
v = torch.zeros_like(init, device=device)
t = 0
batch = 0
epoch = 1
batch_idx = torch.split(
    torch.arange(n_bundles, device=device), int(batch_size/2))
n_batch = math.ceil(2*n_bundles/batch_size)
loss_t = torch.zeros(max_epoch*n_batch, device=device)
grad_norm = torch.zeros(max_epoch*n_batch, device=device)
delta_f_tr = torch.zeros(max_epoch*n_batch, device=device)
delta_m_tr = torch.zeros(max_epoch*n_batch, device=device)
beta2_tr = torch.zeros(max_epoch*n_batch, device=device)

while epoch <= max_epoch:
    q_f_batch = q_f_all[batch_idx[batch], ...]
    q_m_batch = q_m_all[batch_idx[batch], ...]
    p_f_batch = p_f_all[batch_idx[batch], ...]
    p_m_batch = p_m_all[batch_idx[batch], ...]
    loss = mle(init, q_f_batch, q_m_batch, p_f_batch, p_m_batch)
    loss_t[t] = loss.item()
    loss.backward()

    with torch.no_grad():
        gradient = init.grad
        grad_norm[t] = torch.linalg.norm(gradient)
        t += 1
        m = beta_1*m + (1-beta_1)*gradient
        v = beta_2*v + (1-beta_2)*(gradient**2)
        m_hat = m / (1-beta_1**t)
        v_hat = v / (1-beta_2**t)
        init -= lr*m_hat / (torch.sqrt(v_hat)+epsilon)
        delta_f_tr[t-1] = torch.exp(init[:, param_idx[4]])
        delta_m_tr[t-1] = torch.exp(init[:, param_idx[5]])
        beta2_tr[t-1] = torch.exp(init[:, param_idx[6]])
        if (epoch == max_epoch) & ((batch+1) == n_batch):
            break
    
    init.grad.zero_()
    batch = (batch+1) % n_batch
    if batch == 0:
        if epoch % 100 == 0:
            print(f'Epoch {epoch} done. Loss={loss_t[t-1]:.5f}.',
                  f'Grad={grad_norm[t-1]:.5f}.')
        epoch += 1

        # Generate new imposters
        idx_del_f = torch.randint(
            high=n_purchased, size=(n_bundles, ), device=device)
        idx_del_m = torch.randint(
            high=n_purchased, size=(n_bundles, ), device=device)

        mask_f = torch.ones(
            (n_bundles, n_upc_f), dtype=torch.bool, device=device)
        mask_m = torch.ones(
            (n_bundles, n_upc_m), dtype=torch.bool, device=device)
        mask_f[
            torch.arange(n_bundles, 
                         device=device).repeat_interleave(n_purchased), 
            q_col_f_all.flatten()] = False
        mask_m[
            torch.arange(n_bundles, 
                         device=device).repeat_interleave(n_purchased), 
            q_col_m_all.flatten()] = False

        candidate_f = torch.nonzero(
            mask_f, as_tuple=True)[1].reshape(n_bundles, -1)
        candidate_m = torch.nonzero(
            mask_m, as_tuple=True)[1].reshape(n_bundles, -1)

        selected_f_idx = torch.stack([torch.randperm(
            n_upc_f-n_purchased, 
            device=device)[:n_imposters] for _ in range(n_bundles)])
        selected_m_idx = torch.stack([torch.randperm(
            n_upc_m-n_purchased, 
            device=device)[:n_imposters] for _ in range(n_bundles)])

        selected_f = candidate_f.gather(1, selected_f_idx)
        selected_m = candidate_m.gather(1, selected_m_idx)

        qi_col_f = torch.cat((
            q_col_f_all[torch.arange(
                n_purchased, device=device
            ).repeat(n_bundles, 1) != idx_del_f.view(-1, 1)].view(
                n_bundles, n_purchased-1
            ).unsqueeze_(-1).expand(n_bundles, n_purchased-1, n_imposters), 
            selected_f.view(n_bundles, 1, n_imposters)), dim=1)
        qi_col_m = torch.cat((
            q_col_m_all[torch.arange(
                n_purchased, device=device
            ).repeat(n_bundles, 1) != idx_del_m.view(-1, 1)].view(
                n_bundles, n_purchased-1
            ).unsqueeze_(-1).expand(n_bundles, n_purchased-1, n_imposters), 
            selected_m.view(n_bundles, 1, n_imposters)), dim=1)

        qi_val_f_new = torch.randint(
            low=1, high=q_max+1, size=(n_bundles, 1, n_imposters), 
            device=device).float()
        qi_val_m_new = torch.randint(
            low=1, high=q_max+1, size=(n_bundles, 1, n_imposters), 
            device=device).float()

        qi_val_f = torch.cat((
            q_val_f_all[torch.arange(
                n_purchased, device=device
            ).repeat(n_bundles, 1) != idx_del_f.view(-1, 1)].view(
                n_bundles, n_purchased-1
            ).unsqueeze_(-1).expand(n_bundles, n_purchased-1, n_imposters), 
            qi_val_f_new), dim=1)
        qi_val_m = torch.cat((
            q_val_m_all[torch.arange(
                n_purchased, device=device
            ).repeat(n_bundles, 1) != idx_del_m.view(-1, 1)].view(
                n_bundles, n_purchased-1
            ).unsqueeze_(-1).expand(n_bundles, n_purchased-1, n_imposters), 
            qi_val_m_new), dim=1)

        q_f_all = [q_f]
        q_m_all = [q_m]

        for i in range(n_imposters):
            qi_f = torch.zeros((n_bundles, n_upc_f), device=device)
            qi_m = torch.zeros((n_bundles, n_upc_m), device=device)
            qi_f = qi_f.scatter_(1, qi_col_f[..., i], qi_val_f[..., i])
            qi_m = qi_m.scatter_(1, qi_col_m[..., i], qi_val_m[..., i])

            q_f_all.append(qi_f)
            q_m_all.append(qi_m)

        q_f_all = torch.stack(q_f_all, dim=2)
        q_m_all = torch.stack(q_m_all, dim=2)

# Results -------------------
delta_f_print = delta_f_tr[range(0, max_epoch*n_batch, n_batch)]
delta_m_print = delta_m_tr[range(0, max_epoch*n_batch, n_batch)]
beta2_print = beta2_tr[range(0, max_epoch*n_batch, n_batch)]
beta1_print = (delta_f_print+delta_m_print+2*beta2_print*w) / 2
eta_f_print = (beta1_print-delta_f_print) / (2*beta2_print*w)
eta_m_print = (beta1_print-delta_m_print) / (2*beta2_print*w)

plt.figure(figsize=(7, 4))
plt.plot(range(max_epoch), eta_f_print.numpy(), label=rf'$\eta^f={eta_f}$')
plt.plot(range(max_epoch), eta_m_print.numpy(), label=rf'$\eta^m={eta_m}$')
plt.ylim(0, 1)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Estimates")
plt.legend()
plt.tight_layout()
plt.show()