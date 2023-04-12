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


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvTSMixerModel(nn.Module):
    """
    Module implementing ConvTSMixer for forecasting.

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
        depth: int,
        dim: int,
        patch_size: int,
        kernel_size: int,
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

        self.distr_output = distr_output

        self.conv_mixer = nn.Sequential(
            nn.Conv2d(
                self._number_of_features, dim, kernel_size=patch_size, stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for i in range(depth)
            ],
            nn.AdaptiveAvgPool2d((self.prediction_length, self.input_size)),
        )

        self.args_proj = self.distr_output.get_args_proj(dim)

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
        # [B, C, D], [B, D], [B, D]

        # [B, 1, C, D]
        past_target_scaled = past_target_scaled.unsqueeze(1)  # channel dim

        log_abs_loc = loc.abs().log1p().unsqueeze(1).expand_as(past_target_scaled)
        log_scale = scale.log().unsqueeze(1).expand_as(past_target_scaled)

        # [B, C, F] -> [B, F, C, 1] -> [B, F, C, D]
        past_time_feat = (
            past_time_feat.transpose(2, 1)
            .unsqueeze(-1)
            .repeat_interleave(dim=-1, repeats=self.input_size)
        )

        conv_mixer_input = torch.cat(
            (
                past_target_scaled,
                log_abs_loc,
                log_scale,
                past_time_feat,
            ),
            dim=1,
        )

        # [B, F, C, D] -> [B, F, P, D]
        nn_out = self.conv_mixer(conv_mixer_input)
        nn_out_reshaped = nn_out.transpose(1, -1).transpose(1, 2)
        distr_args = self.args_proj(nn_out_reshaped)

        return distr_args, loc, scale
