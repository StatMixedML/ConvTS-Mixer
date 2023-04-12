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
        - in_channels (int): number of input channels
        - ts_length (int): time series length
        - dropout (float): dropout rate
    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """

    def __init__(self, ts_length, dropout=0.1, batch_norm=True):
        super().__init__()

        # BatchNorm1d is applied to the time dimension
        self.batch_norm = nn.BatchNorm1d(ts_length) if batch_norm is True else None

        # MLP for time embedding
        modules = []
        modules.append(nn.Linear(ts_length, ts_length))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        self.time_mlp = nn.Sequential(*modules)

    def forward(self, x):
        x_norm = x if self.batch_norm is None else self.batch_norm(x)
        x_time = self.time_mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        return x + x_time  # not sure if we need a residual connection here, the paper doesn't mention it.


class MLP_Feat(nn.Module):
    """MLPs for feature embedding.

    :argument
        - in_channels (int): input channels
        - ts_length (int): time series length
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate, default 0.1

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
        modules = []
        modules.append(nn.Linear(in_channels, embed_dim))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        self.feat_mlp1 = nn.Sequential(*modules)

        self.feat_mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = x if self.batch_norm is None else self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x_feat = self.feat_mlp1(x_norm)
        return x + self.feat_mlp2(x_feat)


class TSMixerModel(nn.Module):
    """
    Module implementing TSMixer for forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimensions
        Size of hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    batch_norm
        Whether to apply batch normalization. Default: ``False``.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        scaling: str,
        input_size: int,
        K: int,
        hidden_size: int,
        dropout: float,
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert K > 0

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

        modules = []
        for i in range(K):
            modules.append(
                MLP_Time(
                    context_length, batch_norm=True, dropout=dropout
                )
            )
            modules.append(
                MLP_Feat(
                    input_size,
                    hidden_size,
                    batch_norm=True,
                    dropout=dropout,
                )
            )
        self.nn = nn.Sequential(*modules)
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

        nn_out = self.nn(
            past_target_scaled.unsqueeze(-1)
            if self.input_size == 1
            else past_target_scaled
        )
        nn_out_reshaped = nn_out.reshape(-1, self.prediction_length, self.input_size)
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, loc, scale
