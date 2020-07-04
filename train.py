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


import os
import random
import torch
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributions as distrib
import torch.nn.functional as F

from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm

import args
import helper
import data.base as data
import module.srvp as srvp
import module.utils as utils

# Mixed-precision training packages
torch_amp_imported = True
try:
    from torch.cuda import amp as torch_amp
except ImportError:
    torch_amp_imported = False
apex_amp_imported = True
try:
    from apex import amp as apex_amp
except ImportError:
    apex_amp_imported = False


def train(forward_fn, optimizer, scaler, batch, device, opt):
    """
    Performs an optimization step.

    Parameters
    ----------
    forward_fn : function
        Forward method of the model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer of the model.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for PyTorch's mixed-precision training. Is None if this setting is disabled.
    batch : torch.*.Tensor
        Tensor containing the bach of the optimization step with shape (length, batch, channels, width, height) and
        float values lying in [0, 1].
    device : torch.device
        Device on which operations are performed.
    opt : helper.DotDict
        Contains the training configuration.

    Returns
    -------
    float
        Batch-averaged loss of the performed iteration.
    float
        Batch-averaged negative log likelihood component of the loss of the performed iteration.
    float
        Batch-averaged KL divergence component for the initial condition y_0 of the loss of the performed iteration.
    float
        Batch-averaged KL divergence component for variables z of the loss of the performed iteration.
    """
    # Zero gradients
    optimizer.zero_grad()

    # Data
    x = batch.to(device)
    nt, n = x.shape[0], x.shape[1]

    # Forward (inference)
    x_, y, z, _, q_y_0_params, q_z_params, p_z_params, res = forward_fn(x, nt, dt=1 / opt.n_euler_steps)

    # Loss
    # NLL
    nll = utils.neg_logprob(x_, x, scale=opt.obs_scale).sum()
    # y_0 KL
    q_y_0 = utils.make_normal_from_raw_params(q_y_0_params)
    kl_y_0 = distrib.kl_divergence(q_y_0, distrib.Normal(0, 1)).sum()
    # z KL
    q_z, p_z = utils.make_normal_from_raw_params(q_z_params), utils.make_normal_from_raw_params(p_z_params)
    kl_z = distrib.kl_divergence(q_z, p_z).sum()
    # ELBO
    loss = nll + opt.beta_y * kl_y_0 + opt.beta_z * kl_z
    # L2 regularization of residuals
    if opt.l2_res > 0:
        l2_res = torch.norm(res, p=2, dim=2).sum()
        loss += opt.l2_res * l2_res
    # Batch average
    loss /= n

    # Backward and weight updated
    if opt.torch_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        if opt.apex_amp:
            with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    # Logs
    with torch.no_grad():
        loss = loss.item()
        nll = nll.sum().item() / n
        kl_y_0 = kl_y_0.item() / n
        kl_z = kl_z.item() / n

    return loss, nll, kl_y_0, kl_z


def evaluate(forward_fn, val_loader, device, opt):
    """
    Evaluates the model on a validation dataset on a number of validation batches.

    Parameters
    ----------
    forward_fn : function
        Forward method of the model, which must be in evaluation mode.
    val_loader : torch.utils.data.DataLoader
        Randomized dataloader for a data.base.VideoDataset dataset.
    device : torch.device
        Device on which operations are performed.
    opt : helper.DotDict
        Contains the training configuration.

    Returns
    -------
    float
        Average negative prediction PSNR.
    """
    inf_len = opt.nt_cond
    assert opt.n_iter_test <= len(val_loader)

    n = 0  # Total number of evaluation videos, updated in the validation loop
    global_psnr = 0  # Sum of all computed prediction PSNR
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            # Stop when the given number of iterations have been done
            if j >= opt.n_iter_test:
                break

            # Data
            x = batch.to(device)
            x_inf = x[:inf_len]
            nt = x.shape[0]
            n_b = x.shape[1]
            n += n_b

            # Perform a given number of predictions per video
            all_x = []
            for _ in range(opt.n_samples_test - 1):
                all_x.append(forward_fn(x_inf, nt, dt=1 / opt.n_euler_steps)[0].cpu())
            all_x = torch.stack(all_x)

            # Sort predictions per PSNR and select the closesto one to the ground truth
            all_mse = torch.mean(F.mse_loss(all_x, x.cpu().expand_as(all_x), reduction='none'), dim=[4, 5])
            all_psnr = torch.mean(10 * torch.log10(1 / all_mse), dim=[1, 3])
            _, idx_best = all_psnr.max(0)
            x_ = all_x[idx_best, :, torch.arange(n_b).to(device)].transpose(0, 1).contiguous().to(device)

            # Compute the final PSNR score
            mse = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=[3, 4])
            psnr = 10 * torch.log10(1 / mse)
            global_psnr += psnr[inf_len:].mean().item() * n_b

    # Average by batch
    return -global_psnr / n


def main(opt):
    """
    Trains SRVP and saved the resulting model.

    Parameters
    ----------
    opt : helper.DotDict
        Contains the training configuration.
    """
    ##################################################################################################################
    # Setup
    ##################################################################################################################
    opt.hostname = os.uname()[1]
    # Device handling (CPU, GPU, multi GPU)
    if opt.device is None:
        device = torch.device('cpu')
        opt.n_gpu = 0
    else:
        opt.n_gpu = len(opt.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device[opt.local_rank])
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        # In the case of multi GPU: sets up distributed training
        if opt.n_gpu > 1 or opt.local_rank > 0:
            torch.distributed.init_process_group(backend='nccl')
            # Since we are in distributed mode, divide batch size by the number of GPUs
            assert opt.batch_size % opt.n_gpu == 0
            opt.batch_size = opt.batch_size // opt.n_gpu
    # Seed
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    else:
        assert isinstance(opt.seed, int) and opt.seed > 0
    print(f'Learning on {opt.n_gpu} GPU(s) (seed: {opt.seed})')
    random.seed(opt.seed)
    np.random.seed(opt.seed + opt.local_rank)
    torch.manual_seed(opt.seed)
    # cuDNN
    if opt.n_gpu > 1 or opt.local_rank > 0:
        assert torch.backends.cudnn.enabled
        cudnn.deterministic = True
    # Mixed-precision training
    if opt.torch_amp and not torch_amp_imported:
        raise ImportError('Mixed-precision not supported by this PyTorch version, upgrade PyTorch or use Apex')
    if opt.apex_amp and not apex_amp_imported:
        raise ImportError('Apex not installed (https://github.com/NVIDIA/apex)')

    ##################################################################################################################
    # Data
    ##################################################################################################################
    print('Loading data...')
    # Load data
    dataset = data.load_dataset(opt, train=True)
    trainset = dataset.get_fold('train')
    valset = dataset.get_fold('val')

    # Handle random seed for dataloader workers
    def worker_init_fn(worker_id):
        np.random.seed((opt.seed + itr + opt.local_rank * opt.n_workers + worker_id) % (2**32 - 1))
    # Dataloader
    sampler = None
    shuffle = True
    if opt.n_gpu > 1:
        # Let the distributed sampler shuffle for the distributed case
        sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        shuffle = False
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, collate_fn=data.collate_fn, sampler=sampler,
                              num_workers=opt.n_workers, shuffle=shuffle, drop_last=True, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(valset, batch_size=opt.batch_size_test, collate_fn=data.collate_fn,
                            num_workers=opt.n_workers, shuffle=True, drop_last=True, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    ##################################################################################################################
    # Model
    ##################################################################################################################
    # Buid model
    print('Building model...')
    model = srvp.StochasticLatentResidualVideoPredictor(opt.nx, opt.nc, opt.nf, opt.nhx, opt.ny, opt.nz, opt.skipco,
                                                        opt.nt_inf, opt.nh_inf, opt.nlayers_inf, opt.nh_res,
                                                        opt.nlayers_res, opt.archi)
    model.init(res_gain=opt.res_gain)
    # Make the batch norms in the model synchronized in the distributed case
    if opt.n_gpu > 1:
        if opt.apex_amp:
            from apex.parallel import convert_syncbn_model
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    ##################################################################################################################
    # Optimizer
    ##################################################################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    opt.n_iter = opt.lr_scheduling_burnin + opt.lr_scheduling_n_iter
    lr_sch_n_iter = opt.lr_scheduling_n_iter
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda i: max(0, (lr_sch_n_iter - i) / lr_sch_n_iter))

    ##################################################################################################################
    # Automatic Mixed Precision
    ##################################################################################################################
    scaler = None
    if opt.torch_amp:
        scaler = torch_amp.GradScaler()
    if opt.apex_amp:
        model, optimizer = apex_amp.initialize(model, optimizer, opt_level=opt.amp_opt_lvl,
                                               keep_batchnorm_fp32=opt.keep_batchnorm_fp32)

    ##################################################################################################################
    # Multi GPU
    ##################################################################################################################
    if opt.n_gpu > 1:
        if opt.apex_amp:
            from apex.parallel import DistributedDataParallel
            forward_fn = DistributedDataParallel(model)
        else:
            forward_fn = torch.nn.parallel.DistributedDataParallel(model)
    else:
        forward_fn = model

    ##################################################################################################################
    # Training
    ##################################################################################################################
    cudnn.benchmark = True  # Activate benchmarks to select the fastest algorithms
    assert opt.n_iter > 0
    itr = 0
    finished = False
    # Progress bar
    if opt.local_rank == 0:
        pb = tqdm(total=opt.n_iter, ncols=0)
    # Current and best model evaluations score
    val_metric = None
    best_val_metric = None
    try:
        while not finished:
            if sampler is not None:
                sampler.set_epoch(opt.seed + itr)
            # -------- TRAIN --------
            for batch in train_loader:
                # Stop when the given number of optimization steps have been done
                if itr >= opt.n_iter:
                    finished = True
                    status_code = 0
                    break

                itr += 1
                model.train()
                # Optimization step on batch
                # Allow PyTorch's mixed-precision computations if required while ensuring retrocompatibilty
                with (torch_amp.autocast() if opt.torch_amp else nullcontext()):
                    loss, nll, kl_y_0, kl_z = train(forward_fn, optimizer, scaler, batch, device, opt)

                # Learning rate scheduling
                if itr >= opt.lr_scheduling_burnin:
                    lr_scheduler.step()

                # Evaluation and model saving are performed on the process with local rank zero
                if opt.local_rank == 0:
                    # Evaluation
                    if itr % opt.val_interval == 0:
                        model.eval()
                        val_metric = evaluate(forward_fn, val_loader, device, opt)
                        if best_val_metric is None or best_val_metric > val_metric:
                            best_val_metric = val_metric
                            torch.save(model.state_dict(), os.path.join(opt.save_path, 'model_best.pt'))

                    # Checkpointing
                    if opt.chkpt_interval is not None and itr % opt.chkpt_interval == 0:
                        torch.save(model.state_dict(), os.path.join(opt.save_path, f'model_{itr}.pt'))

                # Progress bar
                if opt.local_rank == 0:
                    pb.set_postfix(loss=loss, disto=nll, rate_y_0=kl_y_0, rate_z=kl_z, val_score=val_metric,
                                   best_val_score=best_val_metric, refresh=False)
                    pb.update()

    except KeyboardInterrupt:
        status_code = 130

    if opt.local_rank == 0:
        pb.close()
    # Save model
    print('Saving...')
    if opt.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'model.pt'))
    print('Done')
    return status_code


if __name__ == '__main__':
    # Arguments
    p = args.create_args()
    # Parse arguments
    opt = helper.DotDict(vars(p.parse_args()))
    # Disable output for all processes but the first one
    if opt.local_rank != 0:
        sys.stdout = open(os.devnull, "w")
    # Main
    main(opt)
