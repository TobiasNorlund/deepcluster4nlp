import torch
from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader
from torchnlp.datasets import imdb_dataset


class ImdbDataset(Dataset):
    def __init__(self, is_train: bool, tokenizer):
        super(ImdbDataset).__init__()
        self.tokenizer = tokenizer
        self.data = imdb_dataset(train=is_train, test=not is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.tokenizer.encode(self.data[index]["text"]))
        }


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataloader(is_train: bool, batch_size=2):
    tokenizer = get_tokenizer()
    imdb_ds = ImdbDataset(is_train, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_loader = DataLoader(imdb_ds, batch_size=batch_size, collate_fn=data_collator)
    return data_loader