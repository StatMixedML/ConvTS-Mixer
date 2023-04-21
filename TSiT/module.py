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

import numpy as np
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


class TSiTModel(nn.Module):
    """
    Module implementing TSiT for forecasting.

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
        nhead: int,
        patch_size: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert depth > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.input_size = input_size
        self.dim = dim
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

        self.conv_proj = nn.Conv2d(
            self._number_of_features, dim, kernel_size=patch_size, stride=patch_size
        )

        self.patch_num = (self.context_length // patch_size) * (
            self.input_size // patch_size
        )

        self.positional_encoding = SinusoidalPositionalEmbedding(self.patch_num, dim)

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth, encoder_norm)

        self.pool = nn.AdaptiveAvgPool2d((self.prediction_length, self.input_size))

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

        proj_input = torch.cat(
            (
                past_target_scaled,
                log_abs_loc,
                log_scale,
                past_time_feat,
            ),
            dim=1,
        )

        x = self.conv_proj(proj_input)
        B, C, H, W = x.shape

        x = x.reshape(B, self.dim, -1)
        x = x.permute(0, 2, 1)  # [B, P, D]
        embed_pos = self.positional_encoding(x.size())
        enc_out = self.encoder(x + embed_pos)

        nn_out = self.pool(enc_out.permute(0, 2, 1).reshape(B, C, H, W))

        # [B, F, C, D] -> [B, F, P, D]

        nn_out_reshaped = nn_out.transpose(1, -1).transpose(1, 2)
        distr_args = self.args_proj(nn_out_reshaped)

        return distr_args, loc, scale


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Features are not interleaved. The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        # set early to avoid an error in pytorch-1.8+
        out.requires_grad = False

        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen x ...]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)
