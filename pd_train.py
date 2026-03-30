import os
from typing import List
import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import odl
from odl.contrib.torch import OperatorModule
from dival.reconstructors.networks.iterative import PrimalDualNet
from dival.measure import PSNR, SSIM, MSE

class LoDoPaBH5Dataset(Dataset):
    def __init__(self, h5_paths_obs: List[str], h5_paths_gt: List[str], obs_key: str = 'data', gt_key: str = 'data'):
        assert len(h5_paths_obs) == len(h5_paths_gt)
        self.h5_obs = h5_paths_obs
        self.h5_gt = h5_paths_gt
        self.obs_key = obs_key
        self.gt_key = gt_key
        self.index_map = []
        for i, (p_obs, p_gt) in enumerate(zip(self.h5_obs, self.h5_gt)):
            with h5py.File(p_obs, 'r') as f_obs, h5py.File(p_gt, 'r') as f_gt:
                n_obs = f_obs[self.obs_key].shape[0]
                n_gt = f_gt[self.gt_key].shape[0]
                n = min(n_obs, n_gt)
                for s in range(n):
                    self.index_map.append((i, s))
    def __len__(self):
        return len(self.index_map)
    def __getitem__(self, idx):
        file_idx, slice_idx = self.index_map[idx]
        with h5py.File(self.h5_obs[file_idx], 'r') as f_obs, h5py.File(self.h5_gt[file_idx], 'r') as f_gt:
            obs = np.array(f_obs[self.obs_key][slice_idx], dtype=np.float32)
            gt = np.array(f_gt[self.gt_key][slice_idx], dtype=np.float32)
        obs = torch.from_numpy(obs).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        return {'obs': obs, 'gt': gt}

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def get_h5_file_paths(dir_path: str) -> list:
    """Get sorted list of h5 file paths in a directory."""
    return sorted([
        os.path.join(dir_path, f) 
        for f in os.listdir(dir_path) if f.endswith('.hdf5') or f.endswith('.h5')
    ])

def to_numpy(img_tensor):
    arr = img_tensor.detach().cpu().squeeze()
    if arr.ndim == 3:
        arr = arr[0]
    return arr.numpy()

def build_primal_dual_model(ray_trafo, device):
    op_mod = OperatorModule(ray_trafo)
    op_adj_mod = OperatorModule(ray_trafo.adjoint)
    model = PrimalDualNet(
        n_iter=10,
        op=op_mod, op_adj=op_adj_mod, op_init=None,
        n_primal=5, n_dual=5,
        use_sigmoid=False, n_layer=3, internal_ch=64, kernel_size=3,
        batch_norm=False, prelu=True, lrelu_coeff=0.2,
    )
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
            torch.nn.init.xavier_uniform_(m.weight)
    model.apply(weights_init)
    return model.to(device)

def train_loop(model, train_loader, val_loader, device, epochs, lr, save_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    first_batch = next(iter(train_loader))
    obs, gt = first_batch['obs'], first_batch['gt']
    print("----- DATA SHAPE CHECK BEFORE TRAINING -----")
    print(f"Observation shape [should be [1, 1000, 513]]: {obs.shape}")
    print(f"Ground Truth shape [should be [1, 362, 362]]: {gt.shape}")
    print(f"Obs min/max: {obs.min().item():.4f}/{obs.max().item():.4f}")
    print(f"GT min/max: {gt.min().item():.4f}/{gt.max().item():.4f}")
    print("----- ODL RAY-TRAFO DEBUG -----")
    print(f"Ray transform domain (should match GT): {model.op.operator.domain.shape}")
    print(f"Ray transform range (should match obs): {model.op.operator.range.shape}")

    best_psnr = -1.0
    best_epoch = -1
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bi, batch in enumerate(tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]")):
            obs = batch['obs'].to(device)
            gt = batch['gt'].to(device)
            if ep == 1 and bi == 0:
                print("Batch [Train] input:", obs.shape, obs.min().item(), obs.max().item())
                print("Batch [Train] target:", gt.shape, gt.min().item(), gt.max().item())
            optimizer.zero_grad()
            recon = model(obs)
            if ep == 1 and bi == 0:
                print("Batch [Train] model output:", recon.shape, recon.min().item(), recon.max().item())
            loss = criterion(recon, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * obs.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        psnr_vals = []
        ssim_vals = []
        mse_vals = []
        mae_vals = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [Val]"):
                obs = batch['obs'].to(device)
                gt = batch['gt'].to(device)
                recon = model(obs)
                val_loss += criterion(recon, gt).item() * obs.size(0)
                for j in range(recon.shape[0]):
                    rec_np = to_numpy(recon[j])
                    gt_np = to_numpy(gt[j])
                    psnr_vals.append(PSNR.apply(rec_np, gt_np))
                    ssim_vals.append(SSIM.apply(rec_np, gt_np))
                    mse_vals.append(MSE.apply(rec_np, gt_np))
                    mae_vals.append(np.mean(np.abs(rec_np - gt_np)))
        val_loss /= len(val_loader.dataset)
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        avg_mse = np.mean(mse_vals)
        avg_mae = np.mean(mae_vals)
        print(f"Epoch {ep}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        print(f"  Metrics - PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, MSE={avg_mse:.6f}, MAE={avg_mae:.6f}")
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = ep
            save_model(model, save_path)
            print(f"  New best model saved (PSNR={best_psnr:.3f}) at epoch {ep}")
    print(f"Training finished. Best PSNR={best_psnr:.3f} (epoch {best_epoch})")

if __name__ == "__main__":
    train_obs_dir = "/DATA/biomedical/observation_train"
    train_gt_dir = "/DATA/biomedical/ground_truth_train"
    val_obs_dir = "/DATA/biomedical/observation_validation"
    val_gt_dir = "/DATA/biomedical/ground_truth_validation"

    train_h5_obs = get_h5_file_paths(train_obs_dir)
    train_h5_gt = get_h5_file_paths(train_gt_dir)
    val_h5_obs = get_h5_file_paths(val_obs_dir)
    val_h5_gt = get_h5_file_paths(val_gt_dir)

    OBS_KEY = 'data'
    GT_KEY = 'data'
    model_save_path = "/DATA/biomedical/best_learned_all_train_pd.pth"
    reco_space = odl.uniform_discr(
        min_pt=[-128, -128], max_pt=[128, 128],
        shape=[362, 362], dtype="float32"
    )
    geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=1000, det_shape=513)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_ds = LoDoPaBH5Dataset(train_h5_obs, train_h5_gt, obs_key=OBS_KEY, gt_key=GT_KEY)
    val_ds = LoDoPaBH5Dataset(val_h5_obs, val_h5_gt, obs_key=OBS_KEY, gt_key=GT_KEY)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    model = build_primal_dual_model(ray_trafo, device)
    train_loop(model, train_loader, val_loader, device=device,
               epochs=10, lr=1e-4, save_path=model_save_path)

