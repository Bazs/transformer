import torch
import torchtext
from attr import define
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

IMDB_DATASET_LEN = 25000


@define
class Params:
    batch_size: int
    train_to_val_ratio: float
    num_workers: int


@define
class DatasetAndLoaders:
    train_loader: DataLoader
    test_loader: DataLoader
    vocab: torchtext.vocab.Vocab


def create_dataloaders(params: Params) -> DatasetAndLoaders:
    """Return train, validation, and test dataloaders."""
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = IMDB(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 1 if x == 2 else 0

    def collate_batch(batch):
        label_list, text_list, masks = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            masks.append(torch.ones(len(processed_text), dtype=torch.int64))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        return text_list, masks, label_list

    train_iter, test_iter = IMDB()

    # Converting to list is highly memory-inefficient, but due to a torchtext bug, we have to do this to shuffle.
    # See https://github.com/pytorch/text/issues/2041
    train_dataset = list(train_iter)
    test_dataset = list(test_iter)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        collate_fn=collate_batch,
        shuffle=True,
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
        test_loader=test_loader,
        vocab=vocab,
    )
