import lightning as L
import torch
import torchmetrics
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from sentiment_classification.models.text_transformer import (
    TransformerForClassification,
)

TRAIN_LOSS_KEY = "train_loss"
VAL_LOSS_KEY = "val_loss"
TRAIN_ACCURACY_KEY = "train_accuracy"
VAL_ACCURACY_KEY = "val_accuracy"


class TransformerLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        if self.trainer.global_step == 0:
            wandb.define_metric(TRAIN_LOSS_KEY, summary="min", goal="minimize")
            wandb.define_metric(TRAIN_ACCURACY_KEY, summary="max", goal="maximize")

        text, mask, label = batch
        predictions = self.model(text, mask=mask).squeeze(1)
        loss = self.loss(predictions, label.float())
        self.log(TRAIN_LOSS_KEY, loss, prog_bar=True, on_epoch=True, on_step=False)

        pred_probs = torch.sigmoid(predictions)
        self.train_accuracy_metric(pred_probs, label.float())
        self.log(TRAIN_ACCURACY_KEY, self.train_accuracy_metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        if self.trainer.global_step == 0:
            wandb.define_metric(VAL_LOSS_KEY, summary="min", goal="minimize")
            wandb.define_metric(VAL_ACCURACY_KEY, summary="max", goal="maximize")

        text, mask, label = batch
        predictions = self.model(text, mask=mask).squeeze(1)
        loss = self.loss(predictions, label.float())
        self.log(VAL_LOSS_KEY, loss, prog_bar=True, on_epoch=True, on_step=False)

        pred_probs = torch.sigmoid(predictions)
        self.val_accuracy_metric(pred_probs, label.float())
        self.log(VAL_ACCURACY_KEY, self.val_accuracy_metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
