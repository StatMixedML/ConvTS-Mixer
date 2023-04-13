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

from typing import Tuple, Optional

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput


class MLP_Time(nn.Module):
    """MLP for time embedding. According to the paper, the authors employ a single layer perceptron.

    :argument
        - ts_length (int): time series length
        - dropout (float): dropout rate
        - batch_norm (bool): whether to apply batch normalization

    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """

    def __init__(self, ts_length, dropout=0.1, batch_norm=True):
        super().__init__()

        # BatchNorm1d is applied to the time dimension
        self.batch_norm = nn.BatchNorm1d(ts_length) if batch_norm is True else None

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(ts_length, ts_length),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = x if self.batch_norm is None else self.batch_norm(x)
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
                 batch_norm = True):
        super().__init__()

        # BatchNorm1d is applied to the feature dimension
        self.batch_norm = nn.BatchNorm1d(in_channels) if batch_norm is True else None

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
        x_norm = x if self.batch_norm is None else self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
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
        self.mlp_time = MLP_Time(ts_length, dropout, batch_norm)
        self.mlp_feat = MLP_Feat(in_channels, embed_dim, dropout, batch_norm)

    def forward(self, x):
        x = self.mlp_time(x)
        x = self.mlp_feat(x)
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
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert n_blocks > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.input_size = input_size

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.distr_output = distr_output

        self.mixer_blocks = nn.Sequential(*[
            Mixer_Block(input_size,
                        context_length,
                        hidden_size,
                        dropout,
                        batch_norm)
            for _ in range(n_blocks)
        ])

        self.fc = nn.Linear(context_length, prediction_length)
        self.args_proj = self.distr_output.get_args_proj(input_size)

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

        past_target_scaled, loc, scale = self.scaler(past_target, past_observed_values)
        past_target_scaled = past_target_scaled.unsqueeze(-1) if self.input_size == 1 else past_target_scaled
        nn_out = self.mixer_blocks(past_target_scaled)
        nn_out = self.fc(nn_out.transpose(1, 2)).transpose(1, 2)
        distr_args = self.args_proj(nn_out)
        return distr_args, loc, scale
