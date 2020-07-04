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


AMP_OPT_LEVELS = ['O0', 'O1', 'O2', 'O3']


ARCH_TYPES = ['dcgan', 'vgg']


DATASETS = ['smmnist', 'kth', 'human', 'bair']


def create_args():
    """
    Creates and returns the argument parser of the training program.

    Returns
    -------
    configargparse.ArgumentParser
    """
    p = configargparse.ArgumentParser(
        prog='Stochastic Latent Residual Video Prediction (training)',
        description='Trains SRVP with the given parameters, and the obtained PyTorch models.',
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )

    # Seed
    p.add('--seed', type=int, metavar='SEED', default=None,
          help='Manual seed. If None, it is chosen randomly.')
    # Save
    p.add('--save_path', type=str, metavar='PATH', required=True,
          help='Path where models should be saved.')

    # Mixed-precision training
    amp_p = p.add_argument_group(title='Mixed-precision training',
                                 description='Choice of mixed-precision training library.')
    amp_p = amp_p.add_mutually_exclusive_group()
    amp_p.add('--torch_amp', action='store_true',
              help='Whether to use PyTorch\'s integrated mixed-precision training.')
    amp_p.add('--apex_amp', action='store_true',
              help='Whether to use Nvidia\'s Apex mixed-precision training.')

    # Apex
    apex_p = p.add_argument_group(title='Apex', description='Apex-related options.')
    apex_p.add('--amp_opt_lvl', type=str, metavar='OPT_LVL', default='O1', choices=AMP_OPT_LEVELS,
               help='Mixed precision optimization level (see Apex documentation).')
    apex_p.add('--keep_batchnorm_fp32', action='store_true', default=None,
               help='Whether to keep batch norm computed on 32 bits (see Apex documentation).')

    # Distributed training
    distr_p = p.add_argument_group(title='Distributed',
                                   description='Options for distributed training (GPU, processes).')
    distr_p.add('--local_rank', type=int, metavar='RANK', default=0,
                help='Local process rank for DistributedDataParallel.')
    distr_p.add('--device', type=int, metavar='DEVICE', default=None, nargs='+',
                help='If not None, indicates the list of GPU indices to use with CUDA, only for training.')
    distr_p.add('--n_workers', type=int, metavar='NB', default=4,
                help='Number of childs processes for data loading.')

    # Experiment configuration
    model_p = p.add_argument_group(title='Model Configuration',
                                   description='Model parameters.')
    model_p.add('--nhx', type=int, metavar='SIZE', default=128,
                help='Size of vectors encoding frames.')
    model_p.add('--ny', type=int, metavar='SIZE', required=True,
                help='Size of the state-space variable (y).')
    model_p.add('--nz', type=int, metavar='SIZE', required=True,
                help='Size of the auxiliary random variable (z).')
    model_p.add('--n_euler_steps', type=int, metavar='STEPS', default=1,
                help='Number of Euler step per frame to perform during training and validation.')
    model_p.add('--nt_inf', type=int, metavar='STEPS', required=True,
                help='Number of time steps used to infer y at t = 1.')
    model_p.add('--obs_scale', type=float, metavar='VAR', default=1,
                help='Standard deviation of the distribution of observations.')
    model_p.add('--archi', type=str, metavar='ARCH', default='dcgan', choices=ARCH_TYPES,
                help='Encoder and decoder architecture.')
    model_p.add('--skipco', action='store_true',
                help='Whether to use skip connections form encoders to decoders.')
    model_p.add('--nf', type=int, metavar='FILTERS', default=64,
                help='Number of filters per image channel in the first encoder and last decoder layer.')
    model_p.add('--nh_res', type=int, metavar='SIZE', default=512,
                help='Size of hidden layers in the temporal model function f.')
    model_p.add('--nlayers_res', type=int, metavar='NB', default=4,
                help='Number of hidden layers in temporal model function f.')
    model_p.add('--nh_inf', type=int, metavar='SIZE', default=256,
                help='Size of hidden layers in inference networks.')
    model_p.add('--nlayers_inf', type=int, metavar='NB', default=3,
                help='Number of hidden layers in inference networks.')
    model_p.add('--res_gain', type=float, metavar='GAIN', default=1.41,
                help='Initialization gain of the linear layers of the MLP in the residual temporal model.')
    opt_p = p.add_argument_group(title='Optimization Configuration',
                                 description='Loss and optimization parameters.')
    opt_p.add('--beta_y', type=float, metavar='BETA', default=1,
              help='Beta scale factor of the KL term for y1 in the loss.')
    opt_p.add('--beta_z', type=float, metavar='BETA', default=1,
              help='Beta scale factor of the KL term for z in the loss.')
    opt_p.add('--l2_res', type=float, metavar='LAMBDA', default=1,
              help='Scale factor for the L2 regularization of residuals in the loss.')
    opt_p.add('--batch_size', type=int, metavar='SIZE', default=128,
              help='Training batch size.')
    opt_p.add('--lr', type=float, metavar='LR', default=0.0003,
              help='Learning rate of Adam optimizer.')
    opt_p.add('--lr_scheduling_burnin', type=int, metavar='STEPS', default=1000000,
              help='Number of optimization steps before decreasing the learning rate.')
    opt_p.add('--lr_scheduling_n_iter', type=int, metavar='STEPS', default=100000,
              help='Number of optimization steps for the linear decay of the learning rate.')

    # Dataset
    data_p = p.add_argument_group(title='Dataset',
                                  description='Chosen dataset and parameters.')
    data_p.add('--dataset', type=str, metavar='DATASET', required=True, choices=DATASETS,
               help='Dataset name.')
    data_p.add('--data_dir', type=str, metavar='DIR', required=True,
               help='Data directory.')
    data_p.add('--seq_len', type=int, metavar='LEN', required=True,
               help='Length of training sequences.')
    data_p.add('--ndigits', type=int, metavar='DIGITS', default=2,
               help='For Moving MNIST only. Number of digits.')
    data_p.add('--max_speed', type=int, metavar='SPEED', default=4,
               help='For Moving MNIST only. Digits maximum speed.')
    data_p.add('--deterministic', action='store_true',
               help='For Moving MNIST only. Whether to consider deterministic, instead of stochastic, bounces.')
    data_p.add('--subsampling', type=int, default=8,
               help='For Human3.6M only. Set the video sampling rate.')
    data_p.add('--nx', type=int, metavar='SIZE', default=64,
               help='Frame size (width and height).')
    data_p.add('--nc', type=int, metavar='CHANNELS', required=True,
               help='Number of color channels in the video (1 for Moving MNIST and KTH, 3 for BAIR and Human3.6M.')

    # Evaluation
    eval_p = p.add_argument_group(title='Evaluation',
                                  description='Evaluation parameters.')
    eval_p.add('--val_interval', type=int, metavar='STEPS', default=5000,
               help='Number of optimization steps between each evaluation (and between each best model saving).')
    eval_p.add('--chkpt_interval', type=int, metavar='STEPS', default=None,
               help='If not None, save intermediate models every specified number of optimization steps.')
    eval_p.add('--batch_size_test', type=int, metavar='SIZE', default=16,
               help='Validation batch size.')
    eval_p.add('--n_iter_test', type=int, metavar='STEPS', default=25,
               help='Number of batch iterations per validation.')
    eval_p.add('--nt_cond', type=int, metavar='STEPS', required=True,
               help='Number of conditioning frames at test time. Must not be smaller that nt_inf.')
    eval_p.add('--n_samples_test', type=int, metavar='NB', default=100,
               help='Number of predictions to perform for the comparison with the ground truth during validation.')

    return p
