import pathlib

import torch
import torchtext
from attr import define
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

IMDB_DATASET_LEN = 25000


@define
class Params:
    batch_size: int
    train_to_val_ratio: float


@define
class DatasetAndLoaders:
    train_loader: DataLoader
    num_train_batches: int
    test_loader: DataLoader
    num_test_batches: int
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
        label_list, text_list, masks = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            masks.append(torch.ones(len(processed_text), dtype=torch.int64))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        )
        return text_list.to(device), masks.to(device), label_list.to(device)

    train_iter, test_iter = IMDB()

    train_loader = DataLoader(
        train_iter,
        batch_size=params.batch_size,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_iter,
        batch_size=params.batch_size,
        collate_fn=collate_batch,
    )
    return DatasetAndLoaders(
        train_loader=train_loader,
        num_train_batches=IMDB_DATASET_LEN // params.batch_size,
        test_loader=test_loader,
        num_test_batches=IMDB_DATASET_LEN // params.batch_size,
        vocab=vocab,
    )
