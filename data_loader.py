import torch
from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader
from torchnlp.datasets import imdb_dataset
from torch.nn.utils.rnn import pad_sequence


class ImdbDataset(Dataset):
    def __init__(self, is_train: bool, tokenizer):
        super(ImdbDataset).__init__()
        self.tokenizer = tokenizer
        self.data = imdb_dataset(train=is_train, test=not is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.tokenizer.encode(self.data[index]["text"])), torch.tensor(0)


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataloader(dataset, tokenizer, batch_size=2, num_workers=1):
    def collate(examples):
        return (pad_sequence([x[0] for x in examples], True, tokenizer.pad_token_id),
                torch.tensor([x[1] for x in examples]))

    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    return data_loader


class ReassignedDataset(Dataset):

    def __init__(self, dataset, pseudolabels):
        self.dataset = dataset
        self.pseudolabels = pseudolabels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], self.pseudolabels[index]
