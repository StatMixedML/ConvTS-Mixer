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
from einops import rearrange

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput

# Only needed for the ablation study
from TSMixer.module import CtxMap, MLPFeatMap


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


class Conv2dPatchMap(nn.Module):
    """
    Conv2d module for the reverse mapping of the patch-tensor.

    Parameters
    ----------
    dim : int
        Dimension of the embeddings.
    patch_size : Tuple[int, int]
        Patch size.
    context_length : int

    prediction_length : int
        Number of time points to predict.
    input_size : int
        Input size.

    Returns
    -------
    x : torch.Tensor
    """
    def __init__(self,
                 dim,
                 patch_size,
                 context_length,
                 prediction_length,
                 input_size):
        super().__init__()
        self.dim = dim
        self.prediction_length = prediction_length
        self.input_size = input_size

        p1 = int(context_length / patch_size[0])
        p2 = int(input_size / patch_size[1])

        stride_h = 1
        k_h = int((prediction_length / stride_h) - p2 + 1)

        stride_w = 1
        k_w = int((input_size / stride_w) - p1 + 1)

        self.conv2d_transpose = nn.ConvTranspose2d(dim, dim, kernel_size=(k_h, k_w), stride=(stride_h, stride_w))

    def forward(self, x):
        x = self.conv2d_transpose(x)
        return x


class Conv1dPatchMap(nn.Module):
    """
    Conv1d module for the reverse mapping of the patch-tensor.

    Parameters
    ----------
    dim : int
        Dimension of the embeddings.
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
    def __init__(self,
                 dim: int,
                 patch_size: int,
                 context_length: int,
                 prediction_length: int,
                 input_size: int):
        super().__init__()
        p1 = int(context_length / patch_size[0])
        p2 = int(input_size / patch_size[1])

        self.dim = dim
        self.prediction_length = prediction_length
        self.input_size = input_size
        self.conv1d = nn.Conv1d(p1*p2*dim, dim*prediction_length*input_size, kernel_size=1, stride=1)


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1).unsqueeze(-1)
        x = self.conv1d(x).reshape(batch_size, self.dim, self.prediction_length, self.input_size)
        return x


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
    def __init__(self,
                 patch_size: int,
                 context_length: int,
                 prediction_length: int,
                 input_size: int):
            super().__init__()
            p1 = int(context_length / patch_size[0])
            p2 = int(input_size / patch_size[1])
            self.fc = nn.Sequential(
                Rearrange("b c w h -> b c (w h)"),
                nn.Linear(p1 * p2, prediction_length * input_size),
                Rearrange("b c (w h) -> b c w h", w=prediction_length, h=input_size),
            )

    def forward(self, x):
        x = self.fc(x)
        return x


def RevMapLayer(layer_type: str,
                pooling_type: str,
                dim: int,
                patch_size: int,
                context_length: int,
                prediction_length: int,
                input_size: int):
    """
    Returns the mapping layer for the reverse mapping of the patch-tensor to [b nf h ns].

    :argument
        layer_type: str = "pooling" or "mlp" or "conv1d"
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
    elif layer_type == "conv1d":
        return Conv1dPatchMap(dim, patch_size, context_length, prediction_length, input_size)
    elif layer_type == "conv2d":
        return Conv2dPatchMap(dim, patch_size, context_length, prediction_length, input_size)
    else:
        raise ValueError("Invalid layer type: {}".format(layer_type))


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
    ablation
        Whether to use the ablation study version of the model. Default: ``False``.
    patch_reverse_mapping_layer
        Type of mapping layer to use for mapping the patch-tensor to [b nf h ns] . Default: ``pooling``.
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
        dropout: float = 0.1,
        expansion_factor_token: float = 0.5,
        expansion_factor: int = 4,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
        ablation: bool = False,
        patch_reverse_mapping_layer: str = "pooling",
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
        self.ablation = ablation

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.distr_output = distr_output

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        if not self.ablation:
            # This model is the original MlpTSMixer model.

            num_patches = (self.context_length // patch_size[0]) * (self.input_size // patch_size[1])

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

        else:
            # This model is the ablation study version of MlpTSMixer. Different from the original model, it uses
            # - MLPs for projecting ctx_len -> pred_len
            # - uses future_time_feat_repeat (=z) already in the mixer-block instead of at the distribution-head
            # - MLPs for feature embedding of x and z

            self.linear_map = CtxMap(self.context_length, self.prediction_length)
            self.mlp_x = MLPFeatMap(self._number_of_features, dim, dropout)
            self.mlp_z = MLPFeatMap(self.num_feat_dynamic_real, dim, dropout)

            # TODO:
            #  - add option for specifying the expansion factor for n_feat_xz*2, instead of using 2 as default
            #  - add option for specifying the expansion_factor_ablation, instead of using expansion_factor/2 as default
            n_feat_xz = dim * 2  # since x and z are concatenated along the feature dimension
            dim_xz = n_feat_xz * 2  # expansion factor for the hidden layer in the patch conv2d
            expansion_factor_ablation = expansion_factor/2
            num_patches = (self.prediction_length // patch_size[0]) * (self.input_size // patch_size[1])

            self.mlp_mixer = nn.Sequential(
                nn.Conv2d(
                    n_feat_xz,
                    dim_xz,
                    kernel_size=patch_size,
                    stride=patch_size
                ),
                Rearrange("b c w h -> b (h w) c"),
                *[
                    nn.Sequential(
                        PreNormResidual(
                            dim_xz,
                            FeedForward(num_patches, expansion_factor_ablation, dropout, chan_first),
                        ),
                        PreNormResidual(
                            dim_xz,
                            FeedForward(dim_xz, expansion_factor_token, dropout, chan_last),
                        ),
                    )
                    for i in range(depth)
                ],
                nn.LayerNorm(dim_xz),
                Rearrange(
                    "b (h w) c -> b c w h",
                    h=int(self.prediction_length / patch_size[0]),
                    w=int(self.input_size / patch_size[1]),
                ),
                RevMapLayer(
                    layer_type=patch_reverse_mapping_layer,
                    pooling_type=pooling_type,
                    dim=dim_xz,
                    patch_size=patch_size,
                    prediction_length=self.prediction_length,
                    context_length=self.prediction_length,
                    input_size=self.input_size,
                ),
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

        # [B, C, D], [B, D], [B, D]
        past_target_scaled, loc, scale = self.scaler(past_target, past_observed_values)
        # [B, 1, C, D]
        past_target_scaled = past_target_scaled.unsqueeze(1)
        log_abs_loc = loc.abs().log1p().unsqueeze(1).expand_as(past_target_scaled)
        log_scale = scale.log().unsqueeze(1).expand_as(past_target_scaled)
        # [B, C, F] -> [B, F, C, 1] -> [B, F, C, D]
        past_time_feat = (
            past_time_feat.transpose(2, 1)
            .unsqueeze(-1)
            .repeat_interleave(dim=-1, repeats=self.input_size)
        )

        if not self.ablation:
            # Original MlpTSMixer

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

        else:
            # Ablation study: MlpTSMixer

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
            nn_out = self.mlp_mixer(y_prime)
            nn_out_reshaped = rearrange(nn_out, "b nf h ns -> b h ns nf")
            distr_args = self.args_proj(nn_out_reshaped)

        return distr_args, loc, scale
