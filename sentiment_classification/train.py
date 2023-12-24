import logging
from pathlib import Path
from attr import define
import cattrs
import hydra
import omegaconf

import torch
from torch import nn, optim
from tqdm import tqdm

from sentiment_classification.dataset import create_dataloaders, Params as DatasetParams
from sentiment_classification.models.utils import save_model_and_optimizer
import datetime

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
            num_batches=dataset_and_loaders.num_train_batches,
            optimizer=optimizer,
            criterion=criterion,
        )
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_model_and_optimizer(
                model=model, optimizer=optimizer, epoch=epoch, filepath=output_dir / f"best_train_epoch_{epoch + 1}.pt"
            )

        valid_loss = evaluate_epoch(
            model,
            data_loader=dataset_and_loaders.test_loader,
            num_batches=dataset_and_loaders.num_test_batches,
            criterion=criterion,
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
    num_batches: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train the model for one epoch."""
    model.train()  # Set the model to training mode

    epoch_loss = 0

    actual_num_batches = 0

    for text_batch, masks_batch, label_batch in tqdm(data_loader, desc="Training batch", total=num_batches):
        actual_num_batches += 1
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass: Compute predictions and loss
        predictions = model(text_batch, mask=masks_batch).squeeze(1)
        loss = criterion(predictions, label_batch.float())

        # Backward pass: compute gradient and update weights
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if actual_num_batches != num_batches:
        _logger.warning(
            "Expected to train on %d batches, but only trained on %d batches", num_batches, actual_num_batches
        )

    return epoch_loss / actual_num_batches


def evaluate_epoch(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, num_batches: int, criterion: nn.Module
) -> float:
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0

    actual_num_batches = 0

    with torch.no_grad():
        for text_batch, masks_batch, label_batch in tqdm(data_loader, desc="Validating batch", total=num_batches):
            actual_num_batches += 1

            predictions = model(text_batch, mask=masks_batch).squeeze(1)
            loss = criterion(predictions, label_batch.float())

            epoch_loss += loss.item()

    if actual_num_batches != num_batches:
        _logger.warning(
            "Expected to validate on %d batches, but only validated on %d batches", num_batches, actual_num_batches
        )

    return epoch_loss / actual_num_batches


def _create_timestamped_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
