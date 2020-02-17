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


import torch

import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F


def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.

    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Activation function `{name}` not yet implemented')


def init_weight(m, init_type='normal', init_gain=0.02):
    """
    Initializes the input module with the given parameters.

    Parameters
    ----------
    m : torch.nn.Module
        Module to initialize.
    init_type : str
        'normal', 'xavier', 'kaiming', or 'orthogonal'. Orthogonal initialization types for convolutions and linear
        operations.
    init_gain : float
        Gain to use for the initialization.
    """
    classname = m.__class__.__name__
    if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'BatchNorm2d':
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distributions with the input mean and eps + softplus(raw scale) * scale_stddev as scale.
    """
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim]
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal


def rsample_normal(raw_params, scale_stddev=1):
    """
    Samples from a normal distribution with given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    """
    normal = make_normal_from_raw_params(raw_params, scale_stddev=scale_stddev)
    sample = normal.rsample()
    return sample


def neg_logprob(raw_params, data, scale=1):
    """
    Computes the negative log density function of a given input with respect to a normal distribution created from the
    input parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    data : torch.Tensor
        Computes the log density function of this tensor.
    scale : float
        Multiplier of the final scale parameter of the Gaussian.
    """
    obs_distrib = distrib.Normal(raw_params, scale)
    return -obs_distrib.log_prob(data)
