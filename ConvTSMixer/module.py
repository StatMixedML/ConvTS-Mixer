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


class MLPPatchMap(nn.Module):
    """
    Module implementing MLPMap for the reverse mapping of the patch-tensor.

    Parameters
    ----------
    patch_size : Tuple[int, int]
        Patch size.
    context_length : int
        Context length.
    prediction_length : int
        Number of time points to predict.
    input_size : int
        Input size.

    Returns
    -------
    x : torch.Tensor
    """

    def __init__(
        self,
        patch_size: Tuple[int, int],
        context_length: int,
        prediction_length: int,
        input_size: int,
    ):
        super().__init__()
        p1 = int(context_length / patch_size[0])
        p2 = int(input_size / patch_size[1])
        self.fc = nn.Sequential(
            Rearrange("b c w h -> b c h w"),
            nn.Linear(p1, prediction_length),
            Rearrange("b c h w -> b c w h"),
            nn.Linear(p2, input_size),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class ConvT2DMap(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Tuple[int, int],
        context_length: int,
        prediction_length: int,
        input_size: int,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.input_size = input_size

        input_context_dim = context_length // patch_size[0]
        input_variate_dim = input_size // patch_size[1]

        stride_context_dim = (
            self._compute_stride(
                input_context_dim,
                prediction_length,
                padding=0,
                kernel_size=patch_size[0],
            )
            + 1
        )
        stride_input_dim = (
            self._compute_stride(
                input_variate_dim,
                input_size,
                padding=0,
                kernel_size=patch_size[1],
            )
            + 1
        )

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=(stride_context_dim, stride_input_dim),
            padding=0,
        )

    def _compute_stride(self, l_in, l_out, padding, kernel_size):
        """
        Method to compute ConvTranspose1d where padding is known.
        """
        return (l_out + 2 * padding - kernel_size) // (l_in - 1)

    def forward(self, x):
        x = self.conv(x)
        return x[..., : self.prediction_length, : self.input_size]


def RevMapLayer(
    layer_type: str,
    pooling_type: str,
    dim: int,
    patch_size: Tuple[int, int],
    context_length: int,
    prediction_length: int,
    input_size: int,
):
    """
    Returns the mapping layer for the reverse mapping of the patch-tensor to [b nf h ns].

    :argument
        layer_type: str = "pooling" or "mlp"
        pooling_type: str = "max" or "mean"
        dim: int = dimension of the embeddings
        patch_size: Tuple[int, int] = patch size
        prediction_length: int = prediction length
        context_length: int = context length
        input_size: int = input size

    :returns
        nn.Module = mapping layer

    """
    if layer_type == "pooling":
        if pooling_type == "max":
            return nn.AdaptiveMaxPool2d((prediction_length, input_size))
        elif pooling_type == "mean":
            return nn.AdaptiveAvgPool2d((prediction_length, input_size))
    elif layer_type == "mlp":
        return MLPPatchMap(patch_size, context_length, prediction_length, input_size)
    elif layer_type == "conv2d-transpose":
        return ConvT2DMap(
            in_channels=dim,
            patch_size=patch_size,
            context_length=context_length,
            prediction_length=prediction_length,
            input_size=input_size,
        )
    else:
        raise ValueError("Invalid layer type: {}".format(layer_type))


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
    pooling_type
        If mapping_layer_type == "pooling", specifies the type of pooling to use for mapping the patch-tensor to
        [b nf h ns] . Default: ``max``.
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
        kernel_size: int,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
        batch_norm: bool = True,
        patch_reverse_mapping_layer: str = "mlp",
        pooling_type: str = "max",
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
            nn.BatchNorm2d(dim)
            if batch_norm
            else nn.LayerNorm(
                [
                    dim,
                    self.context_length // patch_size[0],
                    self.input_size // patch_size[1],
                ]
            ),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                            if batch_norm
                            else nn.LayerNorm(
                                [
                                    dim,
                                    self.context_length // patch_size[0],
                                    self.input_size // patch_size[1],
                                ]
                            ),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                    if batch_norm
                    else nn.LayerNorm(
                        [
                            dim,
                            self.context_length // patch_size[0],
                            self.input_size // patch_size[1],
                        ]
                    ),
                )
                for i in range(depth)
            ],
            RevMapLayer(
                layer_type=patch_reverse_mapping_layer,
                pooling_type=pooling_type,
                dim=dim,
                patch_size=patch_size,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                input_size=self.input_size,
            ),
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

        log_abs_loc = loc.sign().unsqueeze(1).expand_as(past_target_scaled) * loc.abs().log1p().unsqueeze(1).expand_as(past_target_scaled)
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

        future_time_feat_repeat = future_time_feat.unsqueeze(2).repeat_interleave(
            dim=2, repeats=self.input_size
        )
        distr_args = self.args_proj(
            torch.cat((nn_out_reshaped, future_time_feat_repeat), dim=-1)
        )

        return distr_args, loc, scale
