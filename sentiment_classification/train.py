import datetime
import logging
from pathlib import Path

import cattrs
import hydra
import omegaconf
import torch
import torchmetrics
from attr import define
from torch import nn, optim
from tqdm import tqdm

from sentiment_classification.dataset import Params as DatasetParams
from sentiment_classification.dataset import create_dataloaders
from sentiment_classification.models.utils import save_model_and_optimizer

_logger = logging.getLogger(Path(__file__).stem)


@define
class Config:
    imdb_path: Path
    dataset_params: DatasetParams
    model: dict
    learning_rate: float
    num_epochs: int
    output_dir: Path


@hydra.main(config_path="config", config_name="train_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = omegaconf.OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    run_name = _create_timestamped_run_name()
    output_dir = config.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Using device: %s", device)

    dataset_and_loaders = create_dataloaders(device=device, params=config.dataset_params)

    model: nn.Module = hydra.utils.instantiate(config.model, vocab_size=len(dataset_and_loaders.vocab)).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # You can adjust learning rate as needed

    best_train_loss = float("inf")
    best_valid_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(
            model,
            data_loader=dataset_and_loaders.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_model_and_optimizer(
                model=model, optimizer=optimizer, epoch=epoch, filepath=output_dir / f"best_train_epoch_{epoch + 1}.pt"
            )

        valid_loss = evaluate_epoch(
            model,
            data_loader=dataset_and_loaders.test_loader,
            criterion=criterion,
            device=device,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model_and_optimizer(
                model=model, optimizer=optimizer, epoch=epoch, filepath=output_dir / f"best_valid_epoch_{epoch + 1}.pt"
            )

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}")


def train_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""
    model.train()  # Set the model to training mode

    epoch_loss = 0

    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)

    for text_batch, masks_batch, label_batch in tqdm(data_loader, desc="Training batch", total=len(data_loader)):
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass: Compute predictions and loss
        predictions = model(text_batch, mask=masks_batch).squeeze(1)
        label_batch = label_batch.float()
        loss = criterion(predictions, label_batch)

        # Backward pass: compute gradient and update weights
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_probs = torch.sigmoid(predictions)
            accuracy_metric(pred_probs, label_batch)

        epoch_loss += loss.item()

    accuracy = accuracy_metric.compute()
    _logger.info("Train accuracy: %s", accuracy)

    return epoch_loss / len(data_loader)


def evaluate_epoch(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0

    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)

    with torch.no_grad():
        for text_batch, masks_batch, label_batch in tqdm(data_loader, desc="Validating batch", total=len(data_loader)):
            predictions = model(text_batch, mask=masks_batch).squeeze(1)
            loss = criterion(predictions, label_batch.float())

            pred_probs = torch.sigmoid(predictions)
            accuracy_metric(pred_probs, label_batch)

            epoch_loss += loss.item()

    accuracy = accuracy_metric.compute()
    _logger.info("Validation accuracy: %s", accuracy)

    return epoch_loss / len(data_loader)


def _create_timestamped_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
