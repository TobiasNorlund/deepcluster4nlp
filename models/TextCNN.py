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
        # self.classifier = nn.Sequential(
        #     nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins]),
        #     nn.Dropout(dropout_rate)
        # )

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, 4096)

        self.relu = nn.ReLU(inplace=True)
        self.top_layer = nn.Linear(4096, num_class)

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, emb_dim))
        #     , nn.ReLU()
        #     , nn.MaxPool2d(kernel_size=2, stride=2)
        #     , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        #     , nn.ReLU()
        #     , nn.MaxPool2d(kernel_size=2, stride=2)
        #     , nn.Flatten(start_dim=1)
        #     , nn.Linear(in_features=12*4*4, out_features=120)
        #     , nn.ReLU()
        #     , nn.Linear(in_features=120, out_features=emb_dim)
        #     , nn.ReLU()
        # )
        # self.top_layer = nn.Linear(in_features=emb_dim, out_features=num_class)

        # load pretrained embedding in embedding layer.
        # emb_dim = vocab_built.vectors.size()[1]
        # self.embed = nn.Embedding(len(vocab_built), emb_dim)
        # self.embed.weight.data.copy_(vocab_built.vectors)
        # emb_dim = 300
        # self.embed = nn.Embedding(10000, emb_dim)
        # self.embed.weight.data.copy_(torch.tensor(np.ones((10000, 300), np.int32)))
        #
        # # Convolutional Layers with different window size kernels
        # self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        #
        # # Dropout layer
        # self.dropout = nn.Dropout(dropout_rate)
        #
        # # FC layer
        # self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)

    def forward(self, x, **kwargs):  # shape: (batch, 16, 768)
        # input x is already embedding vectors from spaCy
        # set the channel dim to a new dimension that’s just 1
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)
        # x = self.classifier(x)  # (batch, 64, 150, 1)

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
    #
    # def forward(self, x):
    #
    #     x = self.classifier(x)
    #     if self.top_layer:
    #         x = self.top_layer(x)
    #     return x



    # emb_x = self.embed(x)
    # emb_x = emb_x.unsqueeze(1)
    #
    # con_x = [conv(emb_x) for conv in self.convs]
    #
    # pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
    # #
    # fc_x = torch.cat(pool_x, dim=1)
    #
    # fc_x = fc_x.squeeze(-1)
    #
    # fc_x = self.dropout(fc_x)
    # logit = self.fc(fc_x)
    # return logit


def textcnn(tokenizer, dim_channel=300, kernel_wins=None, dropout_rate=0.5, num_class=2):
    if kernel_wins is None:
        kernel_wins = [3, 4, 5]
    model = TextCNN(tokenizer.vocab_size, dim_channel, kernel_wins, dropout_rate, num_class)
    return model
