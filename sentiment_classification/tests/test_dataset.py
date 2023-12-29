import torch

from sentiment_classification.data.dataset import Params, create_dataloaders


def test_create_dataloaders():
    batch_size = 2
    params = Params(batch_size=batch_size, train_to_val_ratio=0.8, num_workers=1)

    datasets = create_dataloaders(params=params)

    for texts, masks, labels in datasets.train_loader:
        assert labels.shape == (batch_size,)
        assert texts.shape[0] == batch_size
        assert masks.shape[0] == batch_size
