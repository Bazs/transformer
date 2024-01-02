from typing import Any, Callable

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

_VAL_LOSS_KEY = "val_loss"


class ImageClassifierLightning(L.LightningModule):
    def __init__(
        self, model_factory: Callable[[], nn.Module], optimizer_factory: Callable, lr_scheduler_factory: Callable
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_factory()
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.loss = nn.BCEWithLogitsLoss()
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        image_batch, label_batch = batch
        logits, _ = self.model(image_batch)
        logits = logits.squeeze(1)
        loss = self.loss(logits, label_batch.float())

        pred_probs = torch.sigmoid(logits)
        self.train_accuracy_metric(pred_probs, label_batch.float())

        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": self.train_accuracy_metric,
                "learning_rate": self.optimizers().param_groups[0]["lr"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        image_batch, label_batch = batch
        logits, _ = self.model(image_batch)
        logits = logits.squeeze(1)
        loss = self.loss(logits, label_batch.float())

        pred_probs = torch.sigmoid(logits)
        self.val_accuracy_metric(pred_probs, label_batch.float())

        self.log_dict(
            {
                _VAL_LOSS_KEY: loss,
                "val_accuracy": self.val_accuracy_metric,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch: tuple, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_batch, _ = batch
        logits, attention = self.model(image_batch)
        logits = logits.squeeze(1)
        pred_probs = torch.sigmoid(logits)
        return pred_probs, attention

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        lr_scheduler = self.lr_scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": _VAL_LOSS_KEY,
            },
        }
