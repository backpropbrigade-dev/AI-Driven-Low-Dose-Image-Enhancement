import os
import h5py
import numpy as np
import torch
import odl
from odl.contrib.torch import OperatorModule
from dival.reconstructors.networks.iterative import PrimalDualNet
from dival.measure import PSNR, SSIM, MSE

def get_h5_file_paths(dir_path: str) -> list:
    return sorted([os.path.join(dir_path, f)
        for f in os.listdir(dir_path) if f.endswith('.hdf5') or f.endswith('.h5')])

def to_numpy(img_tensor):
    arr = img_tensor.detach().cpu().squeeze()
    if arr.ndim == 3:
        arr = arr[0]
    return arr.numpy()

def build_primal_dual_model(ray_trafo, device):
    op_mod = OperatorModule(ray_trafo)
    op_adj_mod = OperatorModule(ray_trafo.adjoint)
    model = PrimalDualNet(
        n_iter=10, op=op_mod, op_adj=op_adj_mod, op_init=None,
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
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    obs_test_dir = "/DATA/biomedical/observation_validation"
    gt_test_dir = "/DATA/biomedical/ground_truth_validation"
    model_path = "/DATA/biomedical/best_learned_all_train_pd.pth"

    obs_test_files = get_h5_file_paths(obs_test_dir)
    gt_test_files = get_h5_file_paths(gt_test_dir)
    assert len(obs_test_files) == len(gt_test_files), "Mismatch in number of test observation/GT files"

    reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128],
                                   shape=[362, 362], dtype="float32")
    geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=1000, det_shape=513)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")

    model = build_primal_dual_model(ray_trafo, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    psnr_list, ssim_list, mse_list, mae_list = [], [], [], []

    for obs_path, gt_path in zip(obs_test_files, gt_test_files):
        with h5py.File(obs_path, 'r') as f_obs, h5py.File(gt_path, 'r') as f_gt:
            n_slices = min(f_obs['data'].shape[0], f_gt['data'].shape[0])
            for slice_index in range(n_slices):
                obs = np.array(f_obs['data'][slice_index], dtype=np.float32)
                gt = np.array(f_gt['data'][slice_index], dtype=np.float32)

                obs_t = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
                gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    recon_t = model(obs_t)
                    rec_np = to_numpy(recon_t[0])
                    gt_np = to_numpy(gt_t[0])
                    psnr_list.append(PSNR.apply(rec_np, gt_np))
                    ssim_list.append(SSIM.apply(rec_np, gt_np))
                    mse_list.append(MSE.apply(rec_np, gt_np))
                    mae_list.append(np.mean(np.abs(rec_np - gt_np)))

        print(f"Completed inference for file: {os.path.basename(obs_path)}")

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)

    results = (
        "========== Validation Mean Metrics ==========\n"
        f"PSNR: {mean_psnr:.4f}\n"
        f"SSIM: {mean_ssim:.4f}\n"
        f"MSE:  {mean_mse:.6f}\n"
        f"MAE:  {mean_mae:.6f}\n"
        f"Num slices evaluated: {len(psnr_list)}\n"
    )

    with open("pd_validation_metrics.txt", "w") as f:
        f.write(results)

    print(results)

