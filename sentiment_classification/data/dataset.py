from typing import Any

import torch
import torchtext
from attr import define
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

from sentiment_classification.data.data_pipeline import (
    CLASSIFICATION_TOKEN,
    get_tokenizer,
    preprocess_text,
)


@define
class Params:
    batch_size: int
    train_to_val_ratio: float
    num_workers: int


@define
class DatasetAndLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab: torchtext.vocab.Vocab


def create_dataloaders(params: Params) -> DatasetAndLoaders:
    """Return train, validation, and test dataloaders."""
    tokenizer = get_tokenizer()

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = IMDB(split="train")
    unknown_token = "<unk>"
    # The classification token is prepended to the text, and is used to predict the sentiment.
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=[unknown_token, CLASSIFICATION_TOKEN])
    vocab.set_default_index(vocab[unknown_token])

    label_pipeline = lambda x: 1 if x == 2 else 0

    def collate_batch(batch):
        label_list, text_list, masks = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = preprocess_text(_text, tokenizer, vocab)
            text_list.append(processed_text)
            masks.append(torch.ones(len(processed_text), dtype=torch.int64))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        return text_list, masks, label_list

    train_iter, test_iter = IMDB()

    # Converting to list is highly memory-inefficient, but due to a torchtext bug we have to do this to shuffle.
    # See https://github.com/pytorch/text/issues/2041
    train_dataset, val_dataset = split_train_val(
        original_train_list=list(train_iter), train_to_val_ratio=params.train_to_val_ratio
    )
    test_dataset = list(test_iter)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=params.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        collate_fn=collate_batch,
        shuffle=False,
        num_workers=params.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        collate_fn=collate_batch,
        shuffle=False,
        num_workers=params.num_workers,
    )
    return DatasetAndLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
    )


def split_train_val(original_train_list: list, train_to_val_ratio: float) -> tuple[list, list]:
    """Split the IMDB dataset into train and validation sets.

    It is assumed that the dataset is not shuffled, and that the positive and negative reviews are grouped together.
    """
    # Get positive and negative reviews separately
    positive_review_indices = [i for i, (label, _) in enumerate(original_train_list) if label == 2]
    negative_review_indices = [i for i, (label, _) in enumerate(original_train_list) if label == 1]

    assert len(positive_review_indices) == len(negative_review_indices)

    num_single_class_train_samples = int(len(positive_review_indices) * train_to_val_ratio)
    positive_train_samples = [original_train_list[i] for i in positive_review_indices[:num_single_class_train_samples]]
    positive_val_samples = [original_train_list[i] for i in positive_review_indices[num_single_class_train_samples:]]
    negative_train_samples = [original_train_list[i] for i in negative_review_indices[:num_single_class_train_samples]]
    negative_val_samples = [original_train_list[i] for i in negative_review_indices[num_single_class_train_samples:]]

    train_list = positive_train_samples + negative_train_samples
    val_list = positive_val_samples + negative_val_samples

    assert len(train_list) + len(val_list) == len(original_train_list)

    return train_list, val_list
