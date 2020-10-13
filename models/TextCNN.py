import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TextCNN(nn.Module):

    def __init__(self, vocab_size, dim_channel, kernel_wins, dropout_rate, num_class):
        super(TextCNN, self).__init__()

        emb_dim = 300
        # todo: need features
        self.features = None

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, 4096)

        self.relu = nn.ReLU(inplace=True)
        self.top_layer = nn.Linear(4096, num_class)

    def forward(self, x, **kwargs):  # shape: (batch, 16, 768)
        # set the channel dim to a new dimension that’s just 1
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)

        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)

        fc_x = self.fc(fc_x)

        if self.top_layer:
            fc_x = self.relu(fc_x)
            fc_x = self.top_layer(fc_x)  # (batch, 64, 75, 1)

        return fc_x

def textcnn(tokenizer, dim_channel=300, kernel_wins=None, dropout_rate=0.5, num_class=2):
    if kernel_wins is None:
        kernel_wins = [3, 4, 5]
    model = TextCNN(tokenizer.vocab_size, dim_channel, kernel_wins, dropout_rate, num_class)
    return model
