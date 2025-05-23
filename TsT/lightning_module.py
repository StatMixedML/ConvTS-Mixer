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

import lightning.pytorch as pl
import torch

from gluonts.core.component import validated

from .module import TsTModel


class TsTLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``TsTModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``TsTModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``TsTModel`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TsTModel(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        distr_args, loc, scale = self.model.forward(*args, **kwargs)
        distr = self.model.distr_output.distribution(distr_args, loc, scale)
        return distr.sample((self.model.num_parallel_samples,)).reshape(
            -1,
            self.model.num_parallel_samples,
            self.model.prediction_length,
            self.model.input_size,
        )

    def _compute_loss(self, batch):
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        target = batch["future_target"]
        observed_target = batch["future_observed_values"]

        assert past_target.shape[1] == self.model.context_length
        assert target.shape[1] == self.model.prediction_length

        distr_args, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=batch["past_time_feat"],
            future_time_feat=batch["future_time_feat"],
        )

        loss_values = self.model.distr_output.loss(
            target=target, distr_args=distr_args, loc=loc, scale=scale
        )
        return (loss_values * observed_target).sum() / torch.maximum(
            torch.tensor(1.0), observed_target.sum()
        )

        # distr = self.model.distr_output.distribution(distr_args, loc, scale)

        # return (self.loss(distr, target) * observed_target).sum() / torch.maximum(
        #     torch.tensor(1.0), observed_target.sum()
        # )

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
