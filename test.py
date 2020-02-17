# Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import configargparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

import helper
import data.base as data
import module.srvp as srvp
from metrics.ssim import ssim_loss
from metrics.lpips.loss import PerceptualLoss
from metrics.fvd.score import fvd as fvd_score


def _ssim_wrapper(sample, gt):
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    ssim = ssim_loss(sample.view(nt * bsz, *img_shape), gt.view(nt * bsz, *img_shape), max_val=1., reduction='none')
    return ssim.mean(dim=[2, 3]).view(nt, bsz, img_shape[0])


def _lpips_wrapper(sample, gt):
    global lpips_model
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    if img_shape[0] == 1:
        sample_ = sample.repeat(1, 1, 3, 1, 1)
        gt_ = gt.repeat(1, 1, 3, 1, 1)
    else:
        sample_ = sample
        gt_ = gt
    lpips = lpips_model(sample_.view(nt * bsz, 3, *img_shape[1:]), gt_.view(nt * bsz, 3, *img_shape[1:]))
    return lpips.view(nt, bsz)


def _get_idx_better(name, ref, hyp):
    if name in ('mse', 'fvd', 'lpips'):
        return (hyp < ref).nonzero().flatten()
    if name in ('psnr', 'ssim'):
        return (hyp > ref).nonzero().flatten()


def _get_idx_worst(name, ref, hyp):
    if name in ('mse', 'fvd', 'lpips'):
        return (hyp > ref).nonzero().flatten()
    if name in ('psnr', 'ssim'):
        return (hyp < ref).nonzero().flatten()


def main(opt):
    """
    Tests SRVP.

    Parameters
    ----------
    opt : DotDict
        Contains the testing configuration.
    """
    ##################################################################################################################
    # Setup
    ##################################################################################################################
    # -- Device handling (CPU, GPU)
    opt.train = False
    if opt.device is None:
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    # Seed
    random.seed(opt.test_seed)
    np.random.seed(opt.test_seed)
    torch.manual_seed(opt.test_seed)
    # cuDNN
    assert torch.backends.cudnn.enabled
    # Load LPIPS model
    global lpips_model
    lpips_model = PerceptualLoss(opt.lpips_dir)

    ##################################################################################################################
    # Load XP config
    ##################################################################################################################
    xp_config = helper.load_json(os.path.join(opt.xp_dir, 'config.json'))
    nt_cond = opt.nt_cond if opt.nt_cond is not None else xp_config.nt_cond
    nt_test = opt.nt_gen if opt.nt_gen is not None else xp_config.seq_len_test

    ##################################################################################################################
    # Load test data
    ##################################################################################################################
    print('Loading data...')
    xp_config.data_dir = opt.data_dir
    xp_config.seq_len = nt_test
    dataset = data.load_dataset(xp_config, train=False)
    testset = dataset.get_fold('test')
    test_loader = DataLoader(testset, batch_size=opt.batch_size, collate_fn=data.collate_fn, pin_memory=True)

    ##################################################################################################################
    # Load model
    ##################################################################################################################
    print('Loading model...')
    model = srvp.StochasticLatentResidualVideoPredictor(xp_config.nx, xp_config.nc, xp_config.nf, xp_config.nhx,
                                                        xp_config.ny, xp_config.nz, xp_config.skipco, xp_config.nt_inf,
                                                        xp_config.nh_inf, xp_config.nlayers_inf, xp_config.nh_res,
                                                        xp_config.nlayers_res, xp_config.archi)
    state_dict = torch.load(os.path.join(opt.xp_dir, 'model.pt'), map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ##################################################################################################################
    # Eval
    ##################################################################################################################
    print('Generating samples...')
    torch.set_grad_enabled(False)
    best_samples = defaultdict(list)
    worst_samples = defaultdict(list)
    results = defaultdict(list)
    cond = []
    cond_rec = []
    gt = []
    random_samples = [[] for _ in range(5)]
    # Evaluation is done by batch
    for batch in tqdm(test_loader, ncols=80, desc='evaluation'):
        # Data
        x = batch.to(device)
        assert nt_test <= len(x)
        x = x[:nt_test]
        x_cond = x[:nt_cond]
        x_target = x[nt_cond:]
        cond.append(x_cond.cpu().mul(255).byte().permute(1, 0, 3, 4, 2))
        gt.append(x_target.cpu().mul(255).byte().permute(1, 0, 3, 4, 2))
        # Predictions
        metric_best = {}
        sample_best = {}
        metric_worst = {}
        sample_worst = {}
        # Encode conditional frames and extracts skip connections
        skip = model.encode(x_cond)[1] if model.skipco != 'none' else None
        # Generate opt.n_samples predictions
        for i in range(opt.n_samples):
            # Infer latent variables
            x_rec, y, _, w, _, _, _, _ = model(x_cond, nt_cond, dt=1 / xp_config.n_euler_steps)
            y_0 = y[-1]
            if i == 0:
                x_rec = x_rec[::xp_config.n_euler_steps]
                cond_rec.append(x_rec.cpu().mul(255).byte().permute(1, 0, 3, 4, 2))
            # Use the model in prediction mode starting from the last inferred state
            y_os = model.generate(y_0, [], nt_test - nt_cond + 1, dt=1 / xp_config.n_euler_steps)[0]
            y = y_os[xp_config.n_euler_steps::xp_config.n_euler_steps].contiguous()
            x_pred = model.decode(w, y, skip).clamp(0, 1)
            # Pixelwise quantitative eval
            mse = torch.mean(F.mse_loss(x_pred, x_target, reduction='none'), dim=[3, 4])
            metrics_batch = {
                'psnr': 10 * torch.log10(1 / mse).mean(2).mean(0).cpu(),
                'ssim': _ssim_wrapper(x_pred, x_target).mean(2).mean(0).cpu(),
                'lpips': _lpips_wrapper(x_pred, x_target).mean(0).cpu()
            }
            x_pred_byte = x_pred.cpu().mul(255).byte().permute(1, 0, 3, 4, 2)
            if i < 5:
                random_samples[i].append(x_pred_byte)
            for name, values in metrics_batch.items():
                if i == 0:
                    metric_best[name] = values.clone()
                    sample_best[name] = x_pred_byte.clone()
                    metric_worst[name] = values.clone()
                    sample_worst[name] = x_pred_byte.clone()
                    continue
                # Best samples
                idx_better = _get_idx_better(name, metric_best[name], values)
                metric_best[name][idx_better] = values[idx_better]
                sample_best[name][idx_better] = x_pred_byte[idx_better]
                # Worst samples
                idx_worst = _get_idx_worst(name, metric_worst[name], values)
                metric_worst[name][idx_worst] = values[idx_worst]
                sample_worst[name][idx_worst] = x_pred_byte[idx_worst]
        # Compute metrics for best samples and register
        for name in sample_best.keys():
            best_samples[name].append(sample_best[name])
            worst_samples[name].append(sample_worst[name])
            results[name].append(metric_best[name])
    # Store best, worst and random samples
    samples = {f'random_{i + 1}': torch.cat(random_sample).numpy() for i, random_sample in enumerate(random_samples)}
    samples['cond_rec'] = torch.cat(cond_rec)
    for name in best_samples.keys():
        samples[f'{name}_best'] = torch.cat(best_samples[name]).numpy()
        samples[f'{name}_worst'] = torch.cat(worst_samples[name]).numpy()
        results[name] = torch.cat(results[name]).numpy()

    ##################################################################################################################
    # Compute FVD
    ##################################################################################################################
    print('Computing FVD...')
    cond = torch.cat(cond, 0).permute(1, 0, 4, 2, 3).float().div(255)
    gt = torch.cat(gt, 0).permute(1, 0, 4, 2, 3).float().div(255)
    ref = torch.cat([cond, gt], 0)
    hyp = torch.from_numpy(samples['random_1']).clone().permute(1, 0, 4, 2, 3).float().div(255)
    hyp = torch.cat([cond, hyp], 0)
    fvd = fvd_score(ref, hyp)

    ##################################################################################################################
    # Print results
    ##################################################################################################################
    print('\n')
    print('Results:')
    for name, res in results.items():
        print(name, res.mean(), '+/-', 1.960 * res.std() / np.sqrt(len(res)))
    print(f'FVD', fvd)

    ##################################################################################################################
    # Save samples
    ##################################################################################################################
    np.savez_compressed(os.path.join(opt.xp_dir, 'results.npz'), **results)
    for name, res in samples.items():
        np.savez_compressed(os.path.join(opt.xp_dir, f'{name}.npz'), samples=res)


if __name__ == '__main__':
    # Arguments
    p = configargparse.ArgParser(prog="Stochastic Latent Residual Video Prediction (testing)",
                                 formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    p.add('--xp_dir', type=str, metavar='DIR', required=True,
          help='Directory where the model configuration file are saved.')
    p.add('--data_dir', type=str, metavar='DIR', required=True,
          help='Directory where the dataset is saved.')
    p.add('--lpips_dir', type=str, metavar='DIR', required=True,
          help='Directory where the LIPS weights are saved.')
    p.add('--nt_cond', type=int, metavar='COND', default=None,
          help='Number of conditioning frames.')
    p.add('--nt_gen', type=int, metavar='GEN', default=None,
          help='Total number of frames (conditioning and predicted frames).')
    p.add('--batch_size', type=int, metavar='BATCH', default=16,
          help='Batch size used to compute metrics.')
    p.add('--n_samples', type=int, metavar='NB_SAMPLES', default=100,
          help='Number of predictions per sequence to produce in order to compute the best PSNR, SSIM and LPIPS.')
    p.add('--device', type=int, metavar='DEVICE', default=None,
          help='GPU where the model should be placed when testing (if None, put it on the CPU)')
    p.add('--test_seed', type=int, metavar='SEED', default=1,
          help='Manual seed.')
    # Parse arguments
    opt = helper.DotDict(vars(p.parse_args()))
    # Main
    main(opt)
