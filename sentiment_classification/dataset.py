import pathlib
from attr import define
import torch
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


@define
class Params:
    batch_size: int
    train_to_val_ratio: float


@define
class DatasetAndLoaders:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    vocab: torchtext.vocab.Vocab


def create_dataloaders(device: torch.device, params: Params) -> DatasetAndLoaders:
    """Return train, validation, and test dataloaders."""
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = IMDB(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 1 if x == "pos" else 0

    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        return label_list.to(device), text_list.to(device), lengths.to(device)

    train_iter, test_iter = IMDB()
    train_dataset = list(train_iter)
    test_dataset = list(test_iter)

    train_len = int(len(train_dataset) * params.train_to_val_ratio)
    valid_len = len(train_dataset) - train_len
    train_dataset, valid_dataset = random_split(train_dataset, [train_len, valid_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    return DatasetAndLoaders(
        train_loader=train_loader, validation_loader=valid_loader, test_loader=test_loader, vocab=vocab
    )
