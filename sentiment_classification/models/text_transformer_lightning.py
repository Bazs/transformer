from typing import Any, Callable

import lightning as L
import torch
import torchmetrics
import torchtext
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

import wandb
from sentiment_classification.models.text_transformer import (
    TransformerForClassification,
)

TRAIN_LOSS_KEY = "train_loss"
VAL_LOSS_KEY = "val_loss"
TRAIN_ACCURACY_KEY = "train_accuracy"
VAL_ACCURACY_KEY = "val_accuracy"
LEARNING_RATE_KEY = "learning_rate"


class TransformerLightningModule(L.LightningModule):
    def __init__(
        self,
        vocab: torchtext.vocab.Vocab,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: callable,
        lr_scheduler_factory: callable,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model_factory()
        self.vocab = vocab
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        if self.trainer.global_step == 0:
            wandb.define_metric(TRAIN_LOSS_KEY, summary="min", goal="minimize")
            wandb.define_metric(TRAIN_ACCURACY_KEY, summary="max", goal="maximize")

        text, mask, label = batch
        predictions, _ = self.model(text, mask=mask)
        predictions = predictions.squeeze(1)
        loss = self.loss(predictions, label.float())
        pred_probs = torch.sigmoid(predictions)

        self.train_accuracy_metric(pred_probs, label.float())

        self.log_dict(
            {
                TRAIN_LOSS_KEY: loss,
                TRAIN_ACCURACY_KEY: self.train_accuracy_metric,
                LEARNING_RATE_KEY: self.optimizers().param_groups[0]["lr"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        if self.trainer.global_step == 0:
            wandb.define_metric(VAL_LOSS_KEY, summary="min", goal="minimize")
            wandb.define_metric(VAL_ACCURACY_KEY, summary="max", goal="maximize")

        text, mask, label = batch
        predictions, _ = self.model(text, mask=mask)
        predictions = predictions.squeeze(1)
        loss = self.loss(predictions, label.float())
        pred_probs = torch.sigmoid(predictions)

        self.val_accuracy_metric(pred_probs, label.float())

        self.log_dict(
            {VAL_LOSS_KEY: loss, VAL_ACCURACY_KEY: self.val_accuracy_metric},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[float, torch.Tensor]:
        """Return the predicted sentiment and the attention scores. Expects a batch containing a single text."""
        if 1 != len(batch):
            raise ValueError("The batch size must be 1 for prediction.")
        text = batch
        mask = torch.ones_like(text)
        predictions, attention = self.model(text, mask=mask)
        predictions = torch.sigmoid(predictions)
        return predictions.item(), attention.squeeze(0)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer_factory(params=self.parameters())
        scheduler = self.lr_scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": VAL_LOSS_KEY,
            },
        }
