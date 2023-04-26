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
from functools import partial

import torch
from torch import nn
from einops.layers.torch import Rearrange

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


class MlpTSMixerModel(nn.Module):
    """
    Module implementing MlpTSMixer for forecasting.

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
        patch_size: Tuple[int, int],
        dropout: float = 0.1,
        expansion_factor_token: float = 0.5,
        expansion_factor: int = 4,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
        max_pool: bool = False,
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

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        num_patches = (self.context_length // patch_size[0]) * (
            self.input_size // patch_size[1]
        )

        self.mlp_mixer = nn.Sequential(
            nn.Conv2d(
                self._number_of_features, dim, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b c w h -> b (h w) c"),
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim,
                        FeedForward(num_patches, expansion_factor, dropout, chan_first),
                    ),
                    PreNormResidual(
                        dim,
                        FeedForward(dim, expansion_factor_token, dropout, chan_last),
                    ),
                )
                for i in range(depth)
            ],
            nn.LayerNorm(dim),
            Rearrange(
                "b (h w) c -> b c w h",
                h=int(self.context_length / patch_size[0]),
                w=int(self.input_size / patch_size[1]),
            ),
            nn.AdaptiveAvgPool2d((self.prediction_length, self.input_size))
            if not max_pool
            else nn.AdaptiveMaxPool2d((self.prediction_length, self.input_size)),
        )

        self.args_proj = self.distr_output.get_args_proj(
            dim + self.num_feat_dynamic_real
        )

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
        nn_out = self.mlp_mixer(conv_mixer_input)
        nn_out_reshaped = nn_out.transpose(1, -1).transpose(1, 2)

        future_time_feat_repeat = future_time_feat.unsqueeze(2).repeat_interleave(
            dim=2, repeats=self.input_size
        )
        distr_args = self.args_proj(
            torch.cat((nn_out_reshaped, future_time_feat_repeat), dim=-1)
        )

        return distr_args, loc, scale
