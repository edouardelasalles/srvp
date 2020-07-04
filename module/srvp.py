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


import math
import torch

import numpy as np
import torch.nn as nn

from functools import partial

from module import conv
from module import utils
from module.mlp import MLP


class StochasticLatentResidualVideoPredictor(nn.Module):
    """
    SRVP model. Please refer to the paper.

    Attributes
    ----------
    nx : int
        Width and height of the video frames.
    nc : int
        Number of channels in the video data.
    ny : int
        Number of dimensions of y (state space variable).
    nz : int
        Number of dimensions of z (auxiliary stochastic variable).
    skipco : bool
        Whether to include skip connections into the decoder.
    nt_inf : int
        Number of timesteps used to infer y_1 and to compute the content variable.
    nh_inf : int
        Size of inference MLP hidden layers.
    nlayers_inf : int
        Number of layers in inference MLPs.
    nh_res : int
        Size of residual MLP hidden layers.
    nlayers_res : int
        Number of layers in residual MLPs.
    nhx : int
        Size of frames encoding (dimension of the encoder output).
    encoder : module.conv.BaseEncoder
        Encoder.
    decoder : module.conv.BaseDecoder
        Decoder.
    w_proj : torch.nn.Module
        Permutation-invariant network (first part of the network computing the content variable).
    w_inf : torch.nn.Module
        Second and last part of the network computing the content variable.
    q_y : module.mlp.MLP
        Inference network for y_1.
    inf_z : torch.nn.LSTM
        LSTM used for the inference of z.
    q_z : torch.nn.Linear
        Inference network for z, given the output of inf_z.
    p_z : module.mlp.MLP
        Prior network for z.
    dynamics : module.mlp.MLP
        Residual MLP function for the dynamics computation.
    """
    def __init__(self, nx, nc, nf, nhx, ny, nz, skipco, nt_inf, nh_inf, nlayers_inf, nh_res, nlayers_res, archi):
        """
        Parameters
        ----------
        nx : int
            Width and height of the video frames.
        nc : int
            Number of channels in the video data.
        nf : int
            Number of filters per channel in the first convolution of the encoder.
        nhx : int
            Size of frames encoding (dimension of the encoder output).
        ny : int
            Number of dimensions of y (state space variable).
        nz : int
            Number of dimensions of z (auxiliary stochastic variable).
        skipco : bool
            Whether to include skip connections into the decoder.
        nt_inf : int
            Number of timesteps used to infer y_1 and to compute the content variable.
        nh_inf : int
            Size of inference MLP hidden layers.
        nlayers_inf : int
            Number of layers in inference MLPs.
        nh_res : int
            Size of residual MLP hidden layers.
        nlayers_res : int
            Number of layers in residual MLPs.
        archi : str
            'dcgan' or 'vgg'. Name of the architecture to use for the encoder and the decoder.
        """
        super().__init__()

        # Attributes
        self.nx = nx
        self.nc = nc
        self.ny = ny
        self.nz = nz
        self.skipco = skipco
        self.nt_inf = nt_inf
        self.nh_inf = nh_inf
        self.nlayers_inf = nlayers_inf
        self.nh_res = nh_res
        self.nlayers_res = nlayers_res
        self.nhx = nhx

        # Modules
        # -- Encoder and decoder
        self.encoder = conv.encoder_factory(archi, self.nx, self.nc, self.nhx, nf)
        self.decoder = conv.decoder_factory(archi, self.nx, self.nc, self.nh_inf + self.ny, nf, self.skipco)
        # -- Content
        self.w_proj = nn.Sequential(nn.Linear(self.nhx, self.nh_inf), nn.ReLU(inplace=True))
        self.w_inf = nn.Sequential(nn.Linear(self.nh_inf, self.nh_inf), nn.Tanh())
        # -- Inference of y
        self.q_y = MLP(self.nhx * self.nt_inf, self.nh_inf, self.ny * 2, self.nlayers_inf)
        # -- Inference of z
        self.inf_z = nn.LSTM(self.nhx, self.nh_inf, 1)
        self.q_z = nn.Linear(self.nh_inf, self.nz * 2)
        # -- Prior
        self.p_z = MLP(self.ny, self.nh_res, self.nz * 2, self.nlayers_res)
        # -- Prediction
        self.dynamics = MLP(self.ny + self.nz, self.nh_res, self.ny, self.nlayers_res)  # Residual function

    def init(self, res_gain=1.41):
        """
        Initializes the networks of the model.

        Parameters
        ----------
        res_gain : float
            Initialization gain to use for the residual function.
        """
        # Initialize encoder and decoder
        init_encdec_weights = partial(utils.init_weight, init_type='normal', init_gain=0.02)
        self.encoder.apply(init_encdec_weights)
        self.decoder.apply(init_encdec_weights)
        # Initialize residual function
        init_res_weights = partial(utils.init_weight, init_type='orthogonal', init_gain=res_gain)
        self.dynamics.apply(init_res_weights)

    def encode(self, x):
        """
        Frame-wise encoding of a sequence of images. Returns the encodings and skip connections.

        Parameters
        ----------
        x : torch.*.Tensor
            Video / sequence of images to encode, of shape (length, batch, channels, width, height).

        Returns
        -------
        torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of frames.
        list
            List of torch.*.Tensor representing skip connections. Must be None when skip connections are not allowed.
            Skip connections are extracted from the last frame in testing mode, and from a random frame during
            training.
        """
        nt, bsz, x_shape = x.shape[0], x.shape[1], x.shape[2:]
        # Flatten the temporal dimension (for convolution encoders)
        x_flat = x.view(nt * bsz, *x_shape)
        # Encode
        hx_flat, skips = self.encoder(x_flat, return_skip=True)
        # Reshape with temporal dimension
        hx = hx_flat.view(nt, bsz, self.nhx)
        # Skip connections
        if self.skipco:
            if self.training:
                # When training, take a random frame to compute the skip connections
                t = torch.randint(nt, size=(bsz,)).to(hx.device)
                index = torch.arange(bsz).to(hx.device)
                skips = [s.view(nt, bsz, *s.shape[1:])[t, index] for s in skips]
            else:
                # When testing, choose the last frame
                skips = [s.view(nt, bsz, *s.shape[1:])[-1] for s in skips]
        else:
            skips = None
        return hx, skips

    def decode(self, w, y, skip):
        """
        Decodes a sequence of state variables y along with a content variable w, and skip connections.

        Parameters
        ----------
        w : torch.*.Tensor
            Content variable of shape (batch, nh_inf).
        y : torch.*.Tensor
            Tensor representing a sequence of state variables, of shape (length, batch, ny).
        skip : list
            List of torch.*.Tensor representing skip connections. Must be None when skip connections are not allowed.

        Returns
        -------
        torch.*.Tensor
            Output sequence of frames, of shape (length, batch, channels, width, height).
        """
        nt, bsz = y.shape[0], y.shape[1]
        # Flatten the temporal dimension (for convolutional decoder)
        y_flat = y.view(nt * bsz, self.ny)
        # Repeat the content variable for each state variable
        w_flat = w.repeat(nt, 1, 1).view(nt * bsz, self.nh_inf)
        # Decoder input
        dec_inp = torch.cat([w_flat, y_flat], 1)
        # Skip connection
        if skip is not None:
            skip = [s.expand(nt, *s.shape) for s in skip]
            skip = [s.reshape(nt * bsz, *s.shape[2:]) for s in skip]
        # Decode and reshape
        x_flat = self.decoder(dec_inp, skip)
        x_ = x_flat.view(nt, bsz, *x_flat.shape[1:])
        return x_

    def infer_w(self, hx):
        """
        Computes the content variable from the data with a permutation-invariant network.

        Parameters
        ----------
        hx : torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of frames.

        Returns
        -------
        torch.*.Tensor
            Output sequence of frames, of shape (length, batch, channels, width, height).
        """
        nt, bsz = hx.shape[0], hx.shape[1]
        if self.training:
            # When training, pick w conditioning on random frames
            t = torch.stack([torch.randperm(nt)[:self.nt_inf] for _ in range(bsz)], 1).to(hx.device)
            index = torch.arange(bsz).repeat(self.nt_inf, 1).to(hx.device)
            h = hx[t.view(-1), index.view(-1)].view(self.nt_inf, bsz, self.nhx)
        else:
            # Otherwise, choose the last nt_inf random frames
            h = hx[-self.nt_inf:]
        # Permutation-invariant appplication
        h = self.w_proj(h)
        h = h.sum(0)
        w = self.w_inf(h)
        return w

    def infer_y(self, hx):
        """
        Infers y_0 (first state variable) from the data.

        Parameters
        ----------
        hx : torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of conditioning frames used
            to infer y_0.

        Returns
        -------
        torch.*.Tensor
            Initial state condition y_0, of shape (batch, ny).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior for the initial state condition y_0, of shape
            (batch, 2 * ny).
        """
        q_y_0_params = self.q_y(hx.permute(1, 0, 2).reshape(hx.shape[1], self.nt_inf * self.nhx))
        y_0 = utils.rsample_normal(q_y_0_params)
        return y_0, q_y_0_params

    def infer_z(self, hx):
        """
        Infers a z variable from the data.

        Parameters
        ----------
        hx : torch.*.Tensor
            Encoding of frame t, of shape (batch, nhx), so that z is inferred from timestep t.

        Returns
        -------
        torch.*.Tensor
            Inferred variable z, of shape (batch, nz).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior of z, of shape (nt - 1, batch, 2 * nz).
        """
        q_z_params = self.q_z(hx)
        z = utils.rsample_normal(q_z_params)
        return z, q_z_params

    def _residual_step(self, y_t, z_tp1, dt):
        """
        Performs a residual / Euler step.

        Parameters
        ----------
        y_t : torch.*.Tensor
            Current state variable, of shape (batch, ny).
        z_tp1 : torch.*.Tensor
            Current auxilizary random variable, of shape (batch, nz).
        dt : float
            Euler stepsize.

        Returns
        -------
        torch.*.Tensor
            Next state y_tp1 computed from the last state y_t and the current z variable z_tp1.
        torch.*.Tensor
            Residual applied to y_t to get the next state y_tp1.
        """
        res_inp = torch.cat([y_t, z_tp1], 1)
        res_tp1 = dt * self.dynamics(res_inp)
        y_tp1 = y_t + res_tp1
        return y_tp1, res_tp1

    def generate(self, y_0, hx, nt, dt, remove_intermediate=True):
        """
        Generates a given number of state vectors (including the input initial condition).

        For timesteps going beyond the frame encodings, generates in prediction mode without inference.

        Parameters
        ----------
        y_0 : torch.*.Tensor
            Initial state vector, of shape (batch, ny).
        hx : torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of input frames.
        nt : int
            Number of latent states to generate corresponding to integer times t, including y_0.
        dt : float
            Euler stepsize. Must be the inverse of an integer.
        remove_intermediate : bool
            If False, returns all computed latent states y. If True, only returns states y corresponding to integer
            times (i.e., ignoring intermediate states obtained by the Euler method).

        Returns
        -------
        torch.*.Tensor
            Tensor representing a sequence of state variables, of shape (length, batch, ny), where length is either
            nt or (nt - 1) / dt + 1, depending on the remove_intermediate argument.
        torch.*.Tensor
            Tensor representing a sequence of variables z, of shape (nt - 1, batch, ny).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior of z, of shape (nt - 1, batch, 2 * nz).
        torch.*.Tensor
            Gaussian parameters of the prior distribution of z, of shape (nt - 1, batch, 2 * nz).
        torch.*.Tensor
            List of all computed residuals, of shape ((nt - 1) / dt, batch, ny).
        """
        # Latent states and prior and posterior distributions storage
        y = [y_0]
        z, q_z_params, p_z_params = [], [], []
        res = []

        # First z inference step (LTSM on frame encodings)
        if hx is not None and len(hx) > 0:
            hx_z = self.inf_z(hx)[0]
        else:
            hx_z = []

        # Oversampling (number of state variable to generate per frame)
        assert (1 / dt).is_integer()
        oversampling = int(1 / dt)

        y_tm1 = y_0  # Previous latent state
        t_data = 0
        # Inference / prediction
        for t in np.linspace(dt, nt - 1, oversampling * (nt - 1)):
            prev_t_data = t_data
            t_data = int(math.ceil(t))  # Next, target timestep
            new_step = t_data != prev_t_data
            if new_step:
                # A new frame is seen and introduces a new z
                p_z_t_params = self.p_z(y_tm1)
                p_z_params.append(p_z_t_params)
                if t_data < len(hx):
                    # If observations are available, we use them to infer z
                    z_t, q_z_t_params = self.infer_z(hx_z[t_data])
                    q_z_params.append(q_z_t_params)
                else:
                    # Otherwise, we are in generation mode, and we sample z_t from the learned prior
                    assert not self.training
                    z_t = utils.rsample_normal(p_z_t_params)
                z.append(z_t)
            else:
                # We reuse the previous z
                z_t = z[-1]
            # Residual step
            y_t, res_t = self._residual_step(y_tm1, z_t, dt)
            # Update previous latent state
            y_tm1 = y_t
            # Register tensors
            if not remove_intermediate or t.is_integer():
                # Only keep latent states corresponding to integer times
                y.append(y_t)
            res.append(res_t)

        # Re-package variables
        y = torch.stack(y)
        z = torch.stack(z) if len(z) > 0 else None
        q_z_params = torch.stack(q_z_params) if len(q_z_params) > 0 else None
        p_z_params = torch.stack(p_z_params) if len(p_z_params) > 0 else None
        res = torch.stack(res)
        return y, z, q_z_params, p_z_params, res

    def forward(self, x, nt, dt, remove_intermediate=True):
        """
        Applies the model.

        Parameters
        ----------
        x : torch.*.Tensor
            Input data of shape (length, batch, channels, width, height) with float values lying in [0, 1].
        nt : int
            Number of frames to generate corresponding to integer times t, starting with the auto-encoding of x[0]
            and including reconstructions.
        dt : float
            Euler stepsize.  Must be the inverse of an integer.
        remove_intermediate : bool
            If False, returns all computed latent states y and frames x. If True, only returns states y and frames x
            corresponding to integer times (i.e., ignoring intermediate frames obtained by the Euler method).

        Returns
        -------
        torch.*.Tensor
            Tensor representing the output video, of shape (length, batch, channels, width, height) with float values
            lying in [0, 1], where length is either nt or (nt - 1) / dt + 1, depending on the remove_intermediate
            argument.
            Its first frames correspond to the reconstruction of the first frames of the input video x (including
            intermediate frames if remove_intermediate is True), and the remaining frames are predictions conditioned
            with x.
        torch.*.Tensor
            Tensor representing a sequence of state variables, of shape (length, batch, ny), where length is either
            nt or (nt - 1) / dt + 1, depending on the remove_intermediate argument.
        torch.*.Tensor
            Tensor representing a sequence of variables z, of shape (nt - 1, batch, ny).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior of z, of shape (nt - 1, batch, 2 * nz).
        torch.*.Tensor
            Output sequence of frames, of shape (length, batch, channels, width, height).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior for the initial state condition y_0, of shape
            (batch, 2 * ny).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior of z, of shape (nt - 1, batch, 2 * nz).
        torch.*.Tensor
            Gaussian parameters of the prior distribution of z, of shape (nt - 1, batch, 2 * nz).
        torch.*.Tensor
            List of all computed residuals, of shape ((nt - 1) / dt, batch, ny).
        """
        # Encode images into vectors, and extract a skip connection
        hx, skipco = self.encode(x)
        # Infer w
        w = self.infer_w(hx)
        # Infer y_0
        y_0, q_y_0_params = self.infer_y(hx[:self.nt_inf])
        # Residual temporal model
        y, z, q_z_params, p_z_params, res = self.generate(y_0, hx, nt, dt, remove_intermediate=remove_intermediate)
        # Decode
        x_ = self.decode(w, y, skipco)
        return x_, y, z, w, q_y_0_params, q_z_params, p_z_params, res
