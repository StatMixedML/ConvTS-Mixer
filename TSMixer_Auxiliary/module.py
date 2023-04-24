# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Tuple, Optional

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput


class MLP_Time(nn.Module):
    """MLP for time embedding. According to the paper, the authors employ a single layer perceptron.

    :argument
        - in_channels (int): input channels
        - ts_length (int): time series length
        - dropout (float): dropout rate
        - batch_norm (bool): whether to apply batch normalization

    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """

    def __init__(self,
                 in_channels: int,
                 ts_length: int,
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        super().__init__()

        # BatchNorm2d is applied to the time dimension
        self.batch_norm2d = nn.BatchNorm2d(ts_length) if batch_norm is True else None
        self.in_channels = in_channels

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(ts_length, ts_length),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.batch_norm2d is not None:
            x_norm = x.unsqueeze(-1) if self.in_channels == 1 else x
            x_norm = self.batch_norm2d(x_norm)
            x_norm = x_norm.squeeze(-1) if self.in_channels == 1 else x_norm
        else:
            x_norm = x
        x_time = self.time_mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        return x + x_time  # not sure if we need a residual connection here, the paper doesn't mention it.


class MLP_Feat(nn.Module):
    """MLPs for feature embedding.

    :argument
        - in_channels (int): input channels
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate, default 0.1
        - batch_norm (bool): whether to apply batch normalization

    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        super().__init__()

        # BatchNorm2d is applied to the feature dimension
        self.batch_norm2d = nn.BatchNorm2d(in_channels) if batch_norm is True else None
        self.in_channels = in_channels

        # MLPs for feature embedding
        self.feat_mlp1 = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feat_mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.batch_norm2d is not None:
            x_norm = x.transpose(1, 2).unsqueeze(-1) if self.in_channels == 1 else x.transpose(1, 2)
            x_norm = self.batch_norm2d(x_norm)
            x_norm = x_norm.transpose(1, 2).squeeze(-1) if self.in_channels == 1 else x_norm.transpose(1, 2)
        else:
            x_norm = x
        x_feat = self.feat_mlp1(x_norm)
        return x + self.feat_mlp2(x_feat)



class Mixer_Block(nn.Module):
    """Mixer block.

    :argument
        - in_channels (int): input channels
        - ts_length (int): time series length
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate, default 0.1
        - batch_norm (bool): whether to apply batch normalization

    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """
    def __init__(self,
                 in_channels: int,
                 ts_length: int,
                 embed_dim: int,
                 dropout: float = 0.1,
                 batch_norm: bool = True):

        super().__init__()
        self.mlp_time = MLP_Time(in_channels, ts_length, dropout, batch_norm)
        self.mlp_feat = MLP_Feat(in_channels, embed_dim, dropout, batch_norm)

    def forward(self, x):
        x = self.mlp_time(x)
        x = self.mlp_feat(x)
        return x


class Mixer(nn.Module):
    """Mixer.

    :argument
        - n_feat (int): number of input features
        - n_static_feat (int): number of static features
        - fcst_h (int): forecast horizon
        - embed_dim (int): embedding dimension
        - num_blocks (int): number of mixer blocks
        - dropout (float): dropout rate, default 0.1
        - batch_norm (bool): whether to apply batch normalization

    :return
        - x (tensor): output tensor of shape (batch_size, fcst_h, embed_dim*2)
    """
    def __init__(self,
                 n_feat: int,
                 n_static_feat: int,
                 fcst_h: int,
                 embed_dim: int,
                 num_blocks: int,
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        super(Mixer, self).__init__()
        self.mixer_blocks = nn.ModuleList([
            Mixer_Block(n_feat,
                        n_static_feat,
                        fcst_h,
                        embed_dim,
                        dropout,
                        batch_norm)
            for _ in range(num_blocks)
        ])

    def forward(self, x, s):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x, s)
        return x



class TSMixerModel(nn.Module):
    """
    Module implementing TSMixer for forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    scaling
        Whether to scale the target values. If "mean", the target values are scaled by the mean of the training set.
        If "std", the target values are scaled by the standard deviation of the training set.
        If "none", the target values are not scaled.
    input_size
        Number of input channels.
    n_blocks
        Number of mixer blocks
    hidden_size
        Size of hidden layers in the feed-forward network.
    dropout
        Dropout rate. Default: ``0.1``.
    batch_norm
        Whether to apply batch normalization.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.

    : References:
        - Algorithm 1 in [TSMixer: An all-MLP Architecture for Time Series Forecasting] (https://arxiv.org/pdf/2303.06053.pdf)
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        scaling: str,
        input_size: int,
        n_blocks: int,
        hidden_size: int,
        dropout: float,
        batch_norm: bool = True,
        distr_output=StudentTOutput(),
        num_feat_static_cat: int = 0,
        cardinality: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert n_blocks > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.input_size = input_size

        self.num_feat_static_cat = num_feat_static_cat

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.distr_output = distr_output

        #  Number of features for sx and sz
        n_feat_sx = hidden_size + input_size
        # n_feat_sz = hidden_size + n_dynamic_feat

        # MLP that maps the length of the input time series to fcst_h
        self.fc_map = nn.Linear(context_length, prediction_length)

        # MLPs, conditioned on static features, that map X and Z to embedding space
        self.mlp_sx = MLP_Feat(num_feat_static_cat, hidden_size, dropout)
        # self.mlp_sz = MLP_Feat(n_static_feat, hidden_size, dropout)
        self.mlp_x = MLP_Feat(n_feat_sx, hidden_size, dropout)
        # self.mlp_z = MLP_Feat(n_feat_sz, hidden_size, dropout)

        # Mixer blocks
        self.mixer_blocks = Mixer(hidden_size * 2, num_feat_static_cat, prediction_length, hidden_size, n_blocks, dropout, batch_norm)

        # MLP that maps the output of the mixer blocks to the output dimension
        self.mlp_out = nn.Linear(hidden_size * 2, hidden_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # MLP that maps the hidden size from self.hidden_map to the distribution output
        self.args_proj = self.distr_output.get_args_proj(hidden_size)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_static_real: Optional[torch.Tensor] = None,
        past_time_feat: Optional[torch.Tensor] = None,
        past_target: Optional[torch.Tensor] = None,
        past_observed_values: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        future_observed_values: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:

        # X: historical data of shape (batch_size, ts_length, Cx)
        # S: static features of shape (batch_size, fcst_h, Cs)

        x = past_target
        s = feat_static_cat

        x, loc, scale = self.scaler(x, past_observed_values)
        x = x.unsqueeze(-1) if self.input_size == 1 else x
        x = self.fc_map(x.transpose(1, 2)).transpose(1, 2)
        x_prime = self.mlp_x(torch.cat([x, self.mlp_sx(s)], dim=2))
        y_prime = x_prime
        y_prime_block = self.mixer_blocks(y_prime, s)
        out = self.layer_norm(self.mlp_out(y_prime_block))
        distr_args = self.args_proj(out)
        return distr_args, loc, scale
