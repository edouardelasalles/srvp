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

from torch.utils.data import DataLoader
from tqdm import tqdm

import args
import helper
import data.base as data
import module.srvp as srvp
import module.utils as utils

try:
    from apex import amp
except ImportError:
    pass


def train(forward_fn, optimizer, batch, device, opt):
    optimizer.zero_grad()
    # data
    x = batch.to(device)
    nt, n = x.shape[0], x.shape[1]
    # forward
    x_, y, z, _, q_y_0_params, q_z_params, p_z_params, res = forward_fn(x, nt, dt=1 / opt.n_euler_steps)
    # loss
    # -- disto
    disto = utils.neg_logprob(x_, x, scale=opt.obs_scale).sum()
    # -- rate y_0
    q_y_0 = utils.make_normal_from_raw_params(q_y_0_params)
    rate_y_0 = distrib.kl_divergence(q_y_0, distrib.Normal(0, 1)).sum()
    # -- rate z
    q_z, p_z = utils.make_normal_from_raw_params(q_z_params), utils.make_normal_from_raw_params(p_z_params)
    rate_z = distrib.kl_divergence(q_z, p_z).sum()
    # -- elbo
    loss = disto + opt.beta_y * rate_y_0 + opt.beta_z * rate_z
    # -- L2 regularization on residue
    if opt.l2_res > 0:
        l2_res = torch.norm(res, p=2, dim=2).sum()
        loss += opt.l2_res * l2_res
    # -- scale
    loss /= n
    # backward
    if opt.amp_opt_lvl != 'none':
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    # update weights
    optimizer.step()
    # log
    with torch.no_grad():
        loss = loss.item()
        disto = disto.sum().item() / n
        rate_y_0 = rate_y_0.item() / n
        rate_z = rate_z.item() / n
    return loss, disto, rate_y_0, rate_z


def main(opt):
    """
    Trains SRVP and saved the resulting model.

    Parameters
    ----------
    opt : DotDict
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
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device[opt.local_rank])
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        # In the case of multi GPU: sets up distributed training
        if opt.n_gpu > 1 or opt.local_rank > 0:
            torch.distributed.init_process_group(backend='nccl')
            # Since we are in distributed mode, divide batch size by the number of GPUs
            assert opt.batch_size % opt.n_gpu == 0
            opt.batch_size = opt.batch_size // opt.n_gpu
    # -- Seed
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    else:
        assert isinstance(opt.seed, int) and opt.seed > 0
    print(f"Learning on {opt.n_gpu} GPU(s) (seed: {opt.seed})")
    random.seed(opt.seed)
    np.random.seed(opt.seed + opt.local_rank)
    torch.manual_seed(opt.seed)
    # -- cuDNN
    if opt.n_gpu > 1 or opt.local_rank > 0:
        assert torch.backends.cudnn.enabled
        cudnn.deterministic = True

    ##################################################################################################################
    # Data
    ##################################################################################################################
    print('Loading data...')
    # Load data
    dataset = data.load_dataset(opt, train=True)
    trainset = dataset.get_fold('train')

    # Handle random seed for dataloader workers
    def worker_init_fn(worker_id):
        np.random.seed((opt.seed + itr + opt.local_rank * 10 + worker_id) % (2**32 - 1))
    # Dataloader
    sampler = None
    shuffle = True
    if opt.n_gpu > 1:
        # Let the distributed sampler shuffle for the distributed case
        sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        shuffle = False
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, collate_fn=data.collate_fn, sampler=sampler,
                              num_workers=opt.num_workers, shuffle=shuffle, drop_last=True, pin_memory=True,
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
        if opt.amp_opt_lvl != 'none':
            try:
                from apex.parallel import convert_syncbn_model
            except ImportError:
                raise ImportError('Please install apex: https://github.com/NVIDIA/apex')
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ##################################################################################################################
    # Optimizer
    ##################################################################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    opt.niter = opt.lr_scheduling_burnin + opt.lr_scheduling_niter
    niter = opt.lr_scheduling_niter
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: max(0, (niter - i) / niter))

    ##################################################################################################################
    # Apex's Automatic Mixed Precision
    ##################################################################################################################
    model.to(device)
    if opt.amp_opt_lvl != 'none':
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex: https://github.com/NVIDIA/apex')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.amp_opt_lvl,
                                          keep_batchnorm_fp32=opt.keep_batchnorm_fp32)

    ##################################################################################################################
    # Multi GPU
    ##################################################################################################################
    if opt.n_gpu > 1:
        if opt.amp_opt_lvl != 'none':
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
    assert opt.niter > 0
    # Progress bar
    if opt.local_rank == 0:
        pb = tqdm(total=opt.niter, ncols=0)
    itr = 0
    finished = False
    try:
        while not finished:
            if sampler is not None:
                sampler.set_epoch(opt.seed + itr)
            # -------- TRAIN --------
            for batch in train_loader:
                # Stop when the given number of optimization steps have been done
                if itr >= opt.niter:
                    finished = True
                    status_code = 0
                    break
                itr += 1
                # Closure
                model.train()
                loss, disto, rate_y_0, rate_z = train(forward_fn, optimizer, batch, device, opt)
                # Learning rate scheduling
                if itr >= opt.lr_scheduling_burnin:
                    lr_scheduler.step()
                # Progress bar
                if opt.local_rank == 0:
                    pb.set_postfix(loss=loss, disto=disto, rate_y_0=rate_y_0, rate_z=rate_z, refresh=False)
                    pb.update()
    except KeyboardInterrupt:
        status_code = 130

    if opt.local_rank == 0:
        pb.close()
    print('Done')
    # Save model
    print('Saving...')
    torch.save(model.state_dict(), os.path.join(opt.save_path, 'model.pt'))
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
