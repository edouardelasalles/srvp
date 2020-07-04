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


import torch.nn as nn

from module import utils


def make_lin_block(n_inp, n_out, activation):
    """
    Creates a linear block formed by an activation function and a linear operation.

    Parameters
    ----------
    n_inp : int
        Input dimension.
    n_out : int
        Output dimension.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function, or no
        activation if 'none' is chosen,  before the linear operation.

    Returns
    -------
    torch.nn.Sequential
        Sequence of the potentially chosen activation function and the input linear block.
    """
    modules = []
    if activation != 'none':
        modules.append(utils.activation_factory(activation))
    modules.append(nn.Linear(n_inp, n_out))
    return nn.Sequential(*modules)


class MLP(nn.Module):
    """
    Module implementing an MLP.
    """
    def __init__(self, n_inp, n_hid, n_out, n_layers, activation='relu'):
        """
        Parameters
        ----------
        n_inp : int
            Input dimension.
        n_hid : int
            Number of dimensions in intermediary layers.
        n_out : int
            Output dimension.
        n_layers : int
            Number of layers in the MLP.
        activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function before every
            linear operation but the first one.
        """
        super().__init__()
        assert n_hid == 0 or n_layers > 1
        modules = [
            make_lin_block(n_inp if il == 0 else n_hid, n_out if il == n_layers - 1 else n_hid,
                           activation if il > 0 else 'none')
            for il in range(n_layers)
        ]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """
        Output of the MLP.

        Parameters
        ----------
        x : torch.*.Tensor
            Input of shape (batch, n_inp).

        Returns
        -------
        torch.*.Tensor
            Output of shape (batch, n_out).
        """
        return self.module(x)
