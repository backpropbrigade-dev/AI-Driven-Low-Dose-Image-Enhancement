"""
LoDoPaB-CT TV Reconstruction for All Validation Slices
- Processes all HDF5 files in ground truth and observation directories
- Handles variable number of slices per file
- Computes TV reconstruction and metrics (PSNR, SSIM, MSE, MAE)
- Aggregates mean metrics across all slices
- Saves mean metrics into a text file
"""

import os
import h5py
import numpy as np
from warnings import warn
from functools import partial
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.nn import MSELoss

import odl
from odl.contrib.torch import OperatorModule
from odl.tomo import fbp_op

from dival.measure import PSNR, SSIM
from dival.util.torch_losses import poisson_loss, tv_loss
from dival.util.constants import MU_MAX

# ============================================================================
# TVAdamCTReconstructor Class
# ============================================================================

class TVAdamCTReconstructor:
    HYPER_PARAMS = {
        'lr': {'default': 1e-3, 'range': [1e-5, 1e-1]},
        'gamma': {'default': 20.56, 'range': [1e-7, 1e-0]},
        'iterations': {'default': 5000, 'range': [1, 50000]},
        'loss_function': {'default': 'mse', 'choices': ['mse', 'poisson']},
        'photons_per_pixel': {'default': 4096},
        'mu_max': {'default': MU_MAX},
        'init_filter_type': {'default': 'Hann'},
        'init_frequency_scaling': {'default': 0.1}
    }

    def __init__(self, ray_trafo, callback_func=None,
                 callback_func_interval=100, show_pbar=True, **kwargs):

        hyper_params = {}
        for key in list(kwargs.keys()):
            if key in self.HYPER_PARAMS:
                hyper_params[key] = kwargs.pop(key)

        self.reco_space = ray_trafo.domain
        self.observation_space = ray_trafo.range

        for key, value in hyper_params.items():
            setattr(self, key, value)
        for key, param_info in self.HYPER_PARAMS.items():
            if not hasattr(self, key):
                setattr(self, key, param_info['default'])

        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.callback_func_interval = callback_func_interval
        self.show_pbar = show_pbar

    def reconstruct(self, observation):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        fbp_operator = fbp_op(
            self.ray_trafo, filter_type=self.init_filter_type,
            frequency_scaling=self.init_frequency_scaling)
        self.output = torch.tensor(np.asarray(fbp_operator(observation)), dtype=torch.float32)[None].to(device)
        self.output.requires_grad = True
        self.optimizer = Adam([self.output], lr=self.lr)

        y_delta = torch.tensor(np.asarray(observation), dtype=torch.float32)
        y_delta = y_delta.view(1, *y_delta.shape).to(device)

        if self.loss_function == 'mse':
            criterion = MSELoss()
        elif self.loss_function == 'poisson':
            criterion = partial(poisson_loss,
                                photons_per_pixel=self.photons_per_pixel,
                                mu_max=self.mu_max)
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.inf
        best_output = self.output.detach().clone()

        for i in tqdm(range(self.iterations),
                      desc='TV Reconstruction', disable=not self.show_pbar):
            self.optimizer.zero_grad()
            loss = criterion(self.ray_trafo_module(self.output),
                             y_delta) + self.gamma * tv_loss(self.output)
            loss.backward()
            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = self.output.detach().clone()

            if self.callback_func is not None and (i % self.callback_func_interval == 0 or i == self.iterations - 1):
                self.callback_func(
                    iteration=i,
                    reconstruction=best_output[0, ...].cpu().numpy(),
                    loss=best_loss)

        return self.reco_space.element(best_output[0, ...].cpu().numpy())

# ============================================================================
# LoDoPaB Geometry Setup
# ============================================================================

def get_lodopab_ray_trafo(im_shape=(362, 362), num_angles=1000, impl='astra_cuda'):
    reco_space = odl.uniform_discr(
        min_pt=[-0.13, -0.13],
        max_pt=[0.13, 0.13],
        shape=im_shape,
        dtype='float32'
    )

    angle_partition = odl.uniform_partition(0, np.pi, num_angles)
    detector_partition = odl.uniform_partition(-0.184, 0.184, 513)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    for backend in [impl, 'astra_cpu', 'astra', 'skimage']:
        try:
            ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=backend)
            print(f"Using ray transform backend: {backend}")
            return ray_trafo
        except Exception as e:
            print(f" Failed to use '{backend}' backend: {e}")

    print("All ASTRA backends unavailable — using 'numpy' implementation (slow)")
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='numpy')
    return ray_trafo

# ============================================================================
# Metric Calculation
# ============================================================================

def calculate_metrics(ground_truth, reconstruction):
    gt = np.asarray(ground_truth, dtype=np.float32)
    recon = np.asarray(reconstruction, dtype=np.float32)

    assert gt.shape == recon.shape, "Ground truth and reconstruction shapes differ!"

    psnr_value = PSNR(gt, recon)
    ssim_value = SSIM(gt, recon)
    mse_value = np.mean((gt - recon) ** 2)
    mae_value = np.mean(np.abs(gt - recon))

    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'mse': mse_value,
        'mae': mae_value
    }

# ============================================================================
# Main Pipeline
# ============================================================================

GROUND_TRUTH_DIR = "/DATA/Nith/ground_truth_validation/"
OBSERVATION_DIR = "/DATA/Nith/observation_validation/"
METRICS_FILE = "tv_validation_metrics.txt"

TV_PARAMS = {
    'lr': 5e-4,
    'gamma': 20.56,
    'iterations': 1000,
    'loss_function': 'poisson',
    'photons_per_pixel': 4096,
    'mu_max': MU_MAX,
    'init_filter_type': 'Hann',
    'init_frequency_scaling': 0.1
}

def list_hdf5_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".hdf5")])

def process_all_files():
    gt_files = list_hdf5_files(GROUND_TRUTH_DIR)
    obs_files = list_hdf5_files(OBSERVATION_DIR)

    assert len(gt_files) == len(obs_files), "Number of ground truth and observation files must match!"

    all_metrics = {'psnr': [], 'ssim': [], 'mse': [], 'mae': []}

    for gt_file, obs_file in zip(gt_files, obs_files):
        print(f"\nProcessing file pair:\n  GT: {gt_file}\n  Obs: {obs_file}")

        with h5py.File(gt_file, 'r') as f_gt, h5py.File(obs_file, 'r') as f_obs:
            gt_key = 'data' if 'data' in f_gt else list(f_gt.keys())[0]
            obs_key = 'data' if 'data' in f_obs else list(f_obs.keys())[0]

            num_slices = min(f_gt[gt_key].shape[0], f_obs[obs_key].shape[0])
            print(f"  Number of slices to process: {num_slices}")

            for slice_idx in tqdm(range(num_slices), desc="Slices"):
                gt_slice = f_gt[gt_key][slice_idx]
                obs_slice = f_obs[obs_key][slice_idx]

                im_shape = gt_slice.shape
                num_angles = obs_slice.shape[0]
                ray_trafo = get_lodopab_ray_trafo(im_shape=im_shape, num_angles=num_angles, impl='astra_cuda')

                observation_odl = ray_trafo.range.element(obs_slice)

                # TV reconstruction
                tv_reconstructor = TVAdamCTReconstructor(ray_trafo, show_pbar=False, **TV_PARAMS)
                tv_image = np.asarray(tv_reconstructor.reconstruct(observation_odl))

                metrics = calculate_metrics(gt_slice, tv_image)
                for key in all_metrics.keys():
                    all_metrics[key].append(metrics[key])

    # Compute mean metrics
    mean_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    print("\nMean metrics across all slices:")
    for key, value in mean_metrics.items():
        print(f"  {key.upper()}: {value:.6f}")

    # Save to text file
    with open(METRICS_FILE, 'w') as f:
        f.write("Mean TV Reconstruction Metrics (all files and slices)\n")
        f.write("="*50 + "\n")
        for key, value in mean_metrics.items():
            f.write(f"{key.upper()}: {value:.6f}\n")
    print(f"\nMetrics saved to {METRICS_FILE}")

if __name__ == "__main__":
    process_all_files()

