import torch

from sentiment_classification.dataset import Params, create_dataloaders
from sentiment_classification.text_transformer import TransformerForClassification, TransformerParams


def test_forward():
    device = torch.device("cpu")

    batch_size = 2
    params = Params(batch_size=batch_size, train_to_val_ratio=0.8)

    datasets = create_dataloaders(device=device, params=params)

    model = TransformerForClassification(
        params=TransformerParams(
            vocab_size=len(datasets.vocab),
            emb_dim=32,
            n_heads=2,
            hid_dim=64,
            n_layers=2,
            output_dim=2,
            dropout=0.1,
            max_seq_length=100,
        )
    )

    for labels, texts, lengths in datasets.train_loader:
        assert labels.shape == (batch_size,)
        assert texts.shape[0] == batch_size
        assert lengths.shape == (batch_size,)

        logits = model(texts, mask=None)
        assert logits.shape == (batch_size, 2)
        break
