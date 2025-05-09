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
from einops.layers.torch import Rearrange
from einops import rearrange

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput


class PreNormResidual(nn.Module):
    """
    Pre-Normalization Residual Block. Applies Layer-Normalization over the features and prediction_length dimensions.

    :argument
        - dim (int): input dimension
        - prediction_length (int): prediction length
        - fn (function): function to be applied

    :return
        - x (tensor): output tensor
    """

    def __init__(self, dim: int, prediction_length: int, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm([dim, prediction_length])

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class CtxMap(nn.Module):
    """
    Module implementing the mapping from the context-length to the forecast length for TSMixer.

    :argument
        - context_length (int): context length
        - prediction_length (int): prediction length

    :return
        - x (tensor): output tensor
    """

    def __init__(self, context_length: int, prediction_length: int):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.fc = nn.Sequential(
            Rearrange("b nf h ns -> b nf ns h"),
            nn.Linear(self.context_length, self.prediction_length),
            Rearrange("b nf ns h -> b nf h ns"),
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class MLPTimeBlock(nn.Module):
    """MLP for time embedding.

    :argument
        - prediction_length (int): prediction length
        - dropout (float): dropout rate

    :return
        - x (tensor): output tensor
    """

    def __init__(self, prediction_length: int, dropout: float = 0.1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(prediction_length, prediction_length),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.time_mlp(x)
        return out


class MLPFeatBlock(nn.Module):
    """MLPs for feature embedding.

    :argument
        - in_channels (int): input channels
        - hidden_size (int): hidden size
        - dropout (float): dropout rate, default 0.1

    :return
        - x (tensor): output tensor
    """

    def __init__(self, in_channels: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()

        self.feat_mlp = nn.Sequential(
            Rearrange("b ns nf h -> b ns h nf"),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, in_channels),
            nn.Dropout(dropout),
            Rearrange("b ns h nf -> b ns nf h"),
        )

    def forward(self, x):
        out = self.feat_mlp(x)
        return out


class MLPFeatMap(nn.Module):
    """MLP on feature domain.

    :argument
        - in_channels (int): input channels
        - hidden_size (int): hidden size
        - dropout (float): dropout rate

    :return
        - x (tensor): output tensor
    """

    def __init__(self, in_channels: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            Rearrange("b nf h ns -> b h ns nf"),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            Rearrange("b h ns nf -> b nf h ns"),
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class TSMixerModel(nn.Module):
    """
    Module implementingTSMixer for forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
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
        depth: int,
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert depth > 0

        self.distr_output = distr_output
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.input_size = input_size
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.linear_map = CtxMap(self.context_length, self.prediction_length)
        self.mlp_x = MLPFeatMap(self._number_of_features, dim, dropout)
        self.mlp_z = MLPFeatMap(self.num_feat_dynamic_real, dim, dropout)

        dim_xz = dim * 2  # since x and z are concatenated along the feature dimension

        self.mlp_mixer_block = nn.Sequential(
            Rearrange("b nf h ns -> b ns nf h"),
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim_xz,
                        self.prediction_length,
                        MLPTimeBlock(self.prediction_length, dropout),
                    ),
                    PreNormResidual(
                        dim_xz,
                        self.prediction_length,
                        MLPFeatBlock(dim_xz, dim_xz * expansion_factor, dropout),
                    ),
                )
                for _ in range(depth)
            ],
            Rearrange("b ns nf h -> b h ns nf"),
        )

        self.args_proj = self.distr_output.get_args_proj(dim_xz)

    @property
    def _number_of_features(self) -> int:
        return (
            self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 3  # 1 + the log(loc) + log1p(scale)
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length, self.input_size),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length, self.input_size),
                    dtype=torch.float,
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

        past_target_scaled = past_target_scaled.unsqueeze(1)  # channel dim

        log_abs_loc = loc.sign().unsqueeze(1).expand_as(past_target_scaled) * loc.abs().log1p().unsqueeze(1).expand_as(past_target_scaled)
        log_scale = scale.log().unsqueeze(1).expand_as(past_target_scaled)

        past_time_feat = (
            past_time_feat.transpose(2, 1)
            .unsqueeze(-1)
            .repeat_interleave(dim=-1, repeats=self.input_size)
        )

        # x: historical data of shape (batch_size, Cx, context_length, n_series)
        # z: future time-varying features of shape (batch_size, Cz, prediction_length, n_series)
        # s: static features of shape (batch_size, Cs, prediction_length, n_series)

        # b: batch
        # h: fcst_h
        # ns: n_series
        # nf: n_features

        x = torch.cat(
            (
                past_target_scaled,
                log_abs_loc,
                log_scale,
                past_time_feat,
            ),
            dim=1,
        )

        future_time_feat_repeat = future_time_feat.unsqueeze(2).repeat_interleave(
            dim=2, repeats=self.input_size
        )

        z = rearrange(future_time_feat_repeat, "b h ns nf -> b nf h ns")

        x = self.linear_map(x)
        x_prime = self.mlp_x(x)
        z_prime = self.mlp_z(z)
        y_prime = torch.cat([x_prime, z_prime], dim=1)
        nn_out = self.mlp_mixer_block(y_prime)  # self.mixer_blocks(y_prime, s)
        distr_args = self.args_proj(nn_out)

        return distr_args, loc, scale
