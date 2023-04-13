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


def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin


class LinearModel(nn.Module):
    """
    Module implementing Linear for forecasting.

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
        hidden_dimensions: Optional[List[int]] = None,
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert hidden_dimensions is None or len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = (
            hidden_dimensions if hidden_dimensions is not None else [20, 20]
        )
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.distr_output = distr_output
        self.batch_norm = batch_norm

        dimensions = [context_length] + self.hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [make_linear_layer(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            make_linear_layer(
                dimensions[-1], prediction_length * self.hidden_dimensions[-1]
            )
        )

        self.nn = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(self.hidden_dimensions[-1])

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
        nn_out = self.nn(past_target_scaled)
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, loc, scale
