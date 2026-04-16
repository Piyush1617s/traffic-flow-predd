import json

nb = {
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}
  },
  "cells": []
}

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":text}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

# 1) Title
nb["cells"].append(md(
"""# GWNet v21 Strong — PEMS08 | Full Parity Notebook

Keeps your original notebook structure while upgrading architecture/training:
- Sparse top-k graph construction
- TemporalMixer + DiffusionGCN wave blocks
- TOD/DOW embeddings + FiLM conditioning
- Multi-scale skip fusion
- Composite loss (MAE+Huber+RMSE), EMA, OneCycleLR
"""
))

# 2) Seed cell
nb["cells"].append(code(
"""import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set: {seed} — reproducible ✓")

set_seed()
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
"""
))

# 3) Config cell
nb["cells"].append(code(
"""class Config:
    data_path    = "/kaggle/input/datasets/piyush1718s/pems08/PEMS08.npz"
    adj_csv_path = "/kaggle/input/datasets/piyush1718s/pems08csv/PEMS08.csv"
    num_nodes    = 170
    in_features  = 3
    seq_len      = 24
    pred_len     = 12
    feature_idx  = 0
    noise_std    = 0.0
    train_ratio  = 0.7
    val_ratio    = 0.1

    # Strong-safe defaults (T4 friendly)
    d_model     = 96
    d_skip      = 320
    d_end       = 384
    d_time      = 48
    n_layers    = 10
    kernel_size = 3
    adp_emb     = 16
    gcn_order   = 2
    n_supports  = 3
    dropout     = 0.10

    topk_graph  = 16
    adj_thres   = 0.12

    batch_size   = 48
    lr           = 1e-3
    warmup_eps   = 5
    epochs       = 70
    patience     = 20
    weight_decay = 1e-4
    best_path    = "gwnet_best.pt"

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GWNet v21 | d={cfg.d_model} d_skip={cfg.d_skip} d_time={cfg.d_time} layers={cfg.n_layers}")
print(f"seq={cfg.seq_len} gcn_order={cfg.gcn_order} batch={cfg.batch_size} | {device}")
"""
))

# 4) Data cell
nb["cells"].append(code(
"""def _topk_sparse_rowwise(A, k):
    N = A.shape[0]
    out = np.zeros_like(A, dtype=np.float32)
    for i in range(N):
        row = A[i]
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[row[idx] > 0]
        out[i, idx] = row[idx]
    return out

def load_pems08(cfg):
    raw  = np.load(cfg.data_path)
    data = raw["data"].astype(np.float32)
    T, N, F = data.shape
    print(f"Shape: {data.shape}")

    mean_np = data.mean(axis=0)
    std_np  = data.std(axis=0) + 1e-8
    data_clean = (data - mean_np[None]) / std_np[None]

    feat_std_raw   = std_np[:, cfg.feature_idx].mean()
    norm_noise_std = cfg.noise_std / (feat_std_raw + 1e-8)
    print("Noise disabled" if cfg.noise_std == 0 else f"Normalised noise σ≈{norm_noise_std:.4f}")

    import pandas as pd, os
    A_dist = None
    if os.path.exists(cfg.adj_csv_path):
        df = pd.read_csv(cfg.adj_csv_path)
        A_raw = np.zeros((N, N), dtype=np.float32)
        for _, r in df.iterrows():
            i, j, c = int(r["from"]), int(r["to"]), float(r["cost"])
            if i < N and j < N:
                A_raw[i, j] = c; A_raw[j, i] = c

        nz = A_raw[A_raw > 0]
        sigma = nz.std() if len(nz) > 0 else 1.0
        A = np.exp(-(A_raw**2)/(sigma**2 + 1e-8))
        np.fill_diagonal(A, 0.0)

        A[A < cfg.adj_thres] = 0.0
        A = _topk_sparse_rowwise(A, cfg.topk_graph)
        A = np.maximum(A, A.T)
        A = A / (A.sum(1, keepdims=True) + 1e-8)
        A_dist = A

        nnz = (A_dist > 0).sum()
        print(f"Sparse adjacency — nnz={nnz} ({nnz/N:.1f} avg degree)")
    else:
        A_dist = np.eye(N, dtype=np.float32)
        print("WARNING: adjacency CSV missing; identity fallback")

    return data_clean, mean_np, std_np, A_dist, norm_noise_std

class TrafficDataset(Dataset):
    def __init__(self, data_clean, seq_len, pred_len, feature_idx,
                 noise_std=0.0, split_start=0, split_end=None, training=False):
        self.data      = data_clean
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.feat_idx  = feature_idx
        self.noise_std = noise_std
        self.training  = training

        T = len(data_clean)
        split_end = split_end if split_end is not None else T
        last_i = split_end - seq_len - pred_len + 1
        self.indices = list(range(split_start, last_i))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        rec = self.data[i:i+self.seq_len].copy()
        y   = self.data[i+self.seq_len:i+self.seq_len+self.pred_len, :, self.feat_idx].copy()

        if self.training and self.noise_std > 0:
            rec += np.random.randn(*rec.shape).astype(np.float32) * self.noise_std

        tod = np.array([(i+t) % 288 for t in range(self.seq_len)], dtype=np.int64)
        dow = np.array([((i+t)//288) % 7 for t in range(self.seq_len)], dtype=np.int64)

        return torch.from_numpy(rec), torch.from_numpy(y), torch.from_numpy(tod), torch.from_numpy(dow)

def build_dataloaders(cfg):
    set_seed()
    data_clean, mean_np, std_np, A_dist, norm_noise = load_pems08(cfg)
    T = len(data_clean)
    t1 = int(T * cfg.train_ratio)
    t2 = int(T * (cfg.train_ratio + cfg.val_ratio))

    ds_kw = dict(data_clean=data_clean, seq_len=cfg.seq_len, pred_len=cfg.pred_len,
                 feature_idx=cfg.feature_idx, noise_std=norm_noise)

    dl_tr = DataLoader(TrafficDataset(**ds_kw, split_start=0,  split_end=t1, training=True),
                       batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    dl_va = DataLoader(TrafficDataset(**ds_kw, split_start=t1, split_end=t2, training=False),
                       batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(TrafficDataset(**ds_kw, split_start=t2, split_end=T,  training=False),
                       batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"Train={len(dl_tr.dataset)} | Val={len(dl_va.dataset)} | Test={len(dl_te.dataset)}")
    return dl_tr, dl_va, dl_te, mean_np, std_np, A_dist, norm_noise

print("Data utilities ready.")
"""
))

# 5) Model blocks
nb["cells"].append(code(
"""class DiffusionGCN(nn.Module):
    def __init__(self, d_in, d_out, n_supports=3, order=2, dropout=0.1):
        super().__init__()
        total_in = d_in * (1 + n_supports * order)
        self.mlp  = nn.Linear(total_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.order = order

    def forward(self, x, supports):  # x: (B*S, N, d)
        hs = [x]
        for A in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("nm,bmd->bnd", A, h)
                hs.append(h)
        return self.drop(self.mlp(torch.cat(hs, dim=-1)))

class WaveBlock(nn.Module):
    def __init__(self, d_model, d_skip, kernel_size, dilation, n_supports, gcn_order, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size),
                                 dilation=(1, dilation), groups=d_model)
        self.pw_f = nn.Conv2d(d_model, d_model, (1,1))
        self.pw_g = nn.Conv2d(d_model, d_model, (1,1))

        self.gcn = DiffusionGCN(d_model, d_model, n_supports, gcn_order, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.skip_conv = nn.Conv2d(d_model, d_skip, (1,1))
        self.res_conv  = nn.Conv2d(d_model, d_model, (1,1))

    def forward(self, x, supports):
        residual = x
        pad = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, [pad, 0])

        t = self.dw_conv(x_pad)
        x = self.drop(torch.tanh(self.pw_f(t)) * torch.sigmoid(self.pw_g(t)))

        B, d, N, S = x.shape
        xg = x.permute(0,3,2,1).reshape(B*S, N, d)
        xg = self.gcn(xg, supports)
        x  = xg.reshape(B, S, N, d).permute(0,3,2,1)

        xn = x.permute(0,2,3,1)  # (B,N,S,d)
        xn = self.norm(xn)
        x  = xn.permute(0,3,1,2)

        skip = self.skip_conv(x)
        x = self.res_conv(x) + residual
        return x, skip

print("Model blocks ready.")
"""
))

# 6) Full model
nb["cells"].append(code(
"""class GWNet(nn.Module):
    def __init__(self, cfg, A_np):
        super().__init__()
        N = cfg.num_nodes

        A_t = torch.FloatTensor(A_np)
        D_fwd = A_t.sum(1, keepdim=True).clamp(min=1e-8)
        D_bwd = A_t.T.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("A_fwd", A_t / D_fwd)
        self.register_buffer("A_bwd", A_t.T / D_bwd)

        self.E1 = nn.Parameter(torch.randn(N, cfg.adp_emb) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, cfg.adp_emb) * 0.01)

        self.start_conv = nn.Conv2d(cfg.in_features, cfg.d_model, (1,1))
        self.node_emb   = nn.Parameter(torch.randn(1, cfg.d_model, N, 1) * 0.01)

        self.tod_emb   = nn.Embedding(288, cfg.d_time)
        self.dow_emb   = nn.Embedding(7, cfg.d_time)
        self.time_proj = nn.Linear(cfg.d_time * 2, cfg.d_model)

        self.film_gamma = nn.Linear(cfg.d_model, cfg.d_model)
        self.film_beta  = nn.Linear(cfg.d_model, cfg.d_model)

        dilations = ([1,2,4,8] * 4)[:cfg.n_layers]
        self.blocks = nn.ModuleList([
            WaveBlock(cfg.d_model, cfg.d_skip, cfg.kernel_size, d,
                      cfg.n_supports, cfg.gcn_order, cfg.dropout)
            for d in dilations
        ])

        self.skip_fuse = nn.Conv2d(cfg.d_skip * 3, cfg.d_skip, (1,1))
        self.end_conv1 = nn.Conv2d(cfg.d_skip, cfg.d_end, (1,1))
        self.end_conv2 = nn.Conv2d(cfg.d_end, cfg.pred_len, (1,1))

    def get_supports(self):
        A_adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)
        return [self.A_fwd, self.A_bwd, A_adp]

    def forward(self, x, _A_dist=None, tod=None, dow=None):
        x = x.permute(0,3,2,1)  # (B,F,N,S)
        x = self.start_conv(x) + self.node_emb

        if tod is not None and dow is not None:
            te = torch.cat([self.tod_emb(tod), self.dow_emb(dow)], dim=-1)
            te = self.time_proj(te)  # (B,S,d_model)
            x  = x + te.permute(0,2,1).unsqueeze(2)

            tm = te.mean(dim=1)
            gamma = self.film_gamma(tm).unsqueeze(-1).unsqueeze(-1)
            beta  = self.film_beta(tm).unsqueeze(-1).unsqueeze(-1)
            x = x * (1 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        supports = self.get_supports()
        skips = []
        for block in self.blocks:
            x, s = block(x, supports)
            skips.append(s)

        s_all = torch.stack(skips, dim=0).sum(dim=0)
        s1 = s_all[:, :, :, -1:]
        s3 = s_all[:, :, :, -3:].mean(-1, keepdim=True)
        s6 = s_all[:, :, :, -6:].mean(-1, keepdim=True)
        s  = self.skip_fuse(torch.cat([s1, s3, s6], dim=1))

        out = F.relu(s)
        out = F.relu(self.end_conv1(out))
        out = self.end_conv2(out)
        return out.squeeze(-1)

print("GWNet v21 defined.")
"""
))

# 7) Metrics/loss
nb["cells"].append(code(
"""def masked_mae(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    return (torch.abs(pred-true)*mask).sum() / (mask.sum()+1e-8)

def huber_loss(pred, true, delta=1.0, null_val=0.0):
    mask = (true != null_val).float()
    err  = torch.abs(pred - true)
    loss = torch.where(err < delta, 0.5 * err**2, delta * (err - 0.5 * delta))
    return (loss * mask).sum() / (mask.sum()+1e-8)

def masked_rmse(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    return torch.sqrt(((pred-true)**2 * mask).sum() / (mask.sum()+1e-8))

def masked_mape(pred, true, low_thresh=10.0):
    mask = (true.abs() > low_thresh).float()
    if mask.sum() < 1:
        return torch.tensor(0.0, device=pred.device)
    return (torch.abs((pred-true)/(true.abs()+1.0))*mask).sum() / mask.sum() * 100

def composite_loss(pred, y):
    return 0.6*masked_mae(pred,y) + 0.3*huber_loss(pred,y) + 0.1*masked_rmse(pred,y)

print("Metrics/loss ready.")
"""
))

# 8) load data + model
nb["cells"].append(code(
"""dl_train, dl_val, dl_test, mean_np, std_np, A_dist_np, norm_noise = build_dataloaders(cfg)
mean_flow = torch.from_numpy(mean_np[:, cfg.feature_idx]).to(device)
std_flow  = torch.from_numpy(std_np[:,  cfg.feature_idx]).to(device)
A_dist    = torch.from_numpy(A_dist_np).to(device)

set_seed()
model = GWNet(cfg, A_dist_np).to(device)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {total:,}")
"""
))

# 9) train utils
nb["cells"].append(code(
"""scaler = torch.amp.GradScaler("cuda")

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model):
        model.load_state_dict(self.backup, strict=True)

@torch.no_grad()
def eval_epoch(model, loader, device, mean_flow, std_flow):
    model.eval()
    maes, rmses, mapes = [], [], []
    for x_rec, y, tod, dow in loader:
        x_rec, y = x_rec.to(device), y.to(device)
        tod, dow = tod.to(device), dow.to(device)
        with torch.amp.autocast("cuda"):
            pred = model(x_rec, None, tod=tod, dow=dow)

        pred_d = pred.float()*std_flow[None,None,:] + mean_flow[None,None,:]
        y_d    = y.float()   *std_flow[None,None,:] + mean_flow[None,None,:]
        maes.append(masked_mae(pred_d, y_d).item())
        rmses.append(masked_rmse(pred_d, y_d).item())
        mapes.append(masked_mape(pred_d, y_d).item())
    return np.mean(maes), np.mean(rmses), np.mean(mapes)

print("Train utils ready.")
"""
))

# 10) train loop
nb["cells"].append(code(
"""set_seed()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
ema = EMA(model, decay=0.999)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr,
    epochs=cfg.epochs,
    steps_per_epoch=len(dl_train),
    pct_start=0.15,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=20.0
)

best_val_mae = float("inf")
patience_cnt = 0
history = {"train_loss":[], "val_mae":[], "val_rmse":[], "val_mape":[]}

print("Baseline → MAE=13.114 | RMSE=22.623 | MAPE=8.471%")
print("="*70)

for epoch in range(1, cfg.epochs+1):
    model.train()
    total = 0.0
    for x_rec, y, tod, dow in dl_train:
        x_rec, y = x_rec.to(device), y.to(device)
        tod, dow = tod.to(device), dow.to(device)

        with torch.amp.autocast("cuda"):
            pred = model(x_rec, None, tod=tod, dow=dow)
            loss = composite_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        total += loss.item()

    train_loss = total / len(dl_train)

    ema.apply_to(model)
    val_mae, val_rmse, val_mape = eval_epoch(model, dl_val, device, mean_flow, std_flow)
    ema.restore(model)

    history["train_loss"].append(train_loss)
    history["val_mae"].append(val_mae)
    history["val_rmse"].append(val_rmse)
    history["val_mape"].append(val_mape)

    tag = ""
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_cnt = 0
        ema.apply_to(model)
        torch.save(model.state_dict(), cfg.best_path)
        ema.restore(model)
        tag = "  ← best ✓"
    else:
        patience_cnt += 1
        if patience_cnt >= cfg.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if epoch % 2 == 0 or tag:
        beat = " 🎯" if (val_mae < 13.114 and val_rmse < 22.623 and val_mape < 8.471) else ""
        print(f"Ep {epoch:03d} | Loss={train_loss:.4f} | MAE={val_mae:.3f} RMSE={val_rmse:.3f} MAPE={val_mape:.2f}%{tag}{beat}")

print(f"Best Val MAE: {best_val_mae:.3f}")
"""
))

# 11) plot
nb["cells"].append(code(
"""fig, axes = plt.subplots(1, 3, figsize=(15,4))
axes[0].plot(history["train_loss"]); axes[0].set_title("Train Loss")
axes[1].plot(history["val_mae"]); axes[1].axhline(13.114, c="r", ls="--"); axes[1].set_title("Val MAE")
axes[2].plot(history["val_rmse"]); axes[2].axhline(22.623, c="r", ls="--"); axes[2].set_title("Val RMSE")
plt.tight_layout(); plt.show()
"""
))

# 12) test
nb["cells"].append(code(
"""model.load_state_dict(torch.load(cfg.best_path, map_location=device))

@torch.no_grad()
def paper_style_eval(model, loader, device, mean_flow, std_flow):
    model.eval()
    all_pred, all_true = [], []
    for x_rec, y, tod, dow in loader:
        x_rec, y = x_rec.to(device), y.to(device)
        tod, dow = tod.to(device), dow.to(device)
        pred = model(x_rec, None, tod=tod, dow=dow)
        pred_d = pred * std_flow[None,None,:] + mean_flow[None,None,:]
        y_d    = y    * std_flow[None,None,:] + mean_flow[None,None,:]
        all_pred.append(pred_d.cpu()); all_true.append(y_d.cpu())

    P = torch.cat(all_pred, 0)
    Y = torch.cat(all_true, 0)
    mae  = torch.abs(P-Y).mean().item()
    rmse = torch.sqrt(((P-Y)**2).mean()).item()
    mask = Y.abs() > 10
    mape = (torch.abs((P[mask]-Y[mask])/(Y[mask].abs()+1.0))).mean().item()*100

    print("="*55)
    print("TEST RESULTS — averaged over all 12 steps")
    print("="*55)
    print(f"MAE  : {mae:.3f}  baseline:13.114  Δ={mae-13.114:+.3f}")
    print(f"RMSE : {rmse:.3f}  baseline:22.623  Δ={rmse-22.623:+.3f}")
    print(f"MAPE : {mape:.3f}% baseline: 8.471% Δ={mape-8.471:+.3f}%")
    print("="*55)
    return mae, rmse, mape

mae, rmse, mape = paper_style_eval(model, dl_test, device, mean_flow, std_flow)
"""
))

# 13) horizon
nb["cells"].append(code(
"""@torch.no_grad()
def horizon_eval(model, loader, device, mean_flow, std_flow):
    model.eval()
    buf = {h:{'mae':[],'rmse':[],'mape':[]} for h in [2,5,11]}
    for x_rec, y, tod, dow in loader:
        x_rec, y = x_rec.to(device), y.to(device)
        tod, dow = tod.to(device), dow.to(device)
        pred = model(x_rec, None, tod=tod, dow=dow)
        pred_d = pred * std_flow[None,None,:] + mean_flow[None,None,:]
        y_d    = y    * std_flow[None,None,:] + mean_flow[None,None,:]

        for h in buf:
            buf[h]['mae'].append(masked_mae(pred_d[:,h,:],  y_d[:,h,:]).item())
            buf[h]['rmse'].append(masked_rmse(pred_d[:,h,:], y_d[:,h,:]).item())
            buf[h]['mape'].append(masked_mape(pred_d[:,h,:], y_d[:,h,:]).item())

    print(f"{'Horizon':>14} | {'MAE':>8} | {'RMSE':>8} | {'MAPE':>9}")
    print("-"*50)
    for h, lbl in zip([2,5,11], ['3-step (15min)','6-step (30min)','12-step (60min)']):
        m = {k:np.mean(v) for k,v in buf[h].items()}
        print(f"{lbl:>14} | {m['mae']:>8.3f} | {m['rmse']:>8.3f} | {m['mape']:>8.2f}%")

horizon_eval(model, dl_test, device, mean_flow, std_flow)
"""
))

with open("gwnet_v21_strong_full.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Saved: gwnet_v21_strong_full.ipynb")