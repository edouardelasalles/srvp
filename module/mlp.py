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


def make_lin_block(ninp, nout, activation):
    """
    Creates a linear block formed by an activation function and a linear operation.

    Parameters
    ----------
    ninp : int
        Input dimension.
    nout : int
        Output dimension.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function before
        the linear operation.
    """
    modules = []
    if activation != 'none':
        modules.append(utils.activation_factory(activation))
    modules.append(nn.Linear(ninp, nout))
    return nn.Sequential(*modules)


class MLP(nn.Module):
    """
    Module implementing an MLP.
    """
    def __init__(self, ninp, nhid, nout, nlayers, activation='relu'):
        """
        Parameters
        ----------
        ninp : int
            Input dimension.
        nhid : int
            Number of dimensions in intermediary layers.
        nout : int
            Output dimension.
        nlayers : int
            Number of layers in the MLP.
        activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function before the
            linear operation.
        """
        super().__init__()
        assert nhid == 0 or nlayers > 1
        modules = [
            make_lin_block(
                ninp=ninp if il == 0 else nhid,
                nout=nout if il == nlayers - 1 else nhid,
                activation=activation if il > 0 else 'none',
            ) for il in range(nlayers)
        ]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)
