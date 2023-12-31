from sentiment_classification.data.dataset import Params, create_dataloaders
from sentiment_classification.models.text_transformer import (
    TransformerForClassification,
    TransformerParams,
)


def test_forward():
    batch_size = 2
    params = Params(batch_size=batch_size, train_to_val_ratio=0.8, num_workers=1)

    datasets = create_dataloaders(params=params)

    model = TransformerForClassification(
        vocab_size=len(datasets.vocab),
        params=TransformerParams(
            emb_dim=32,
            n_heads=2,
            hid_dim=64,
            n_layers=2,
            output_dim=2,
            dropout=0.1,
            max_seq_length=1000,
        ),
    )

    for texts, masks, labels in datasets.train_loader:
        logits, _ = model(texts, mask=masks)
        assert logits.shape == (batch_size, 2)
        break
