import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, vocab_size, dim_channel, kernel_wins, dropout_rate, num_class, embedding_dim, num_class_features=4096):
        super(TextCNN, self).__init__()

        self.num_class_features = num_class_features
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, embedding_dim)) for w in kernel_wins])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class_features)

        self.relu = nn.ReLU(inplace=True)
        self.top_layer = nn.Linear(num_class_features, num_class)

    def forward(self, x, **kwargs):
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
            fc_x = self.top_layer(fc_x)

        return fc_x

    def set_top_layer(self, cluster_list_length):
        # set last fully connected layer
        # cluster_list_length = len(deepcluster.cluster_lists)
        
        self.top_layer = nn.Linear(self.num_class_features, cluster_list_length)
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()
        self.top_layer.cuda()
    
    def reset_top_layer(self):
        self.top_layer = None


def textcnn(tokenizer, dim_channel=300, kernel_wins=None, dropout_rate=0.5, num_class=2, embedding_dim=300, num_class_features=4096):
    if kernel_wins is None:
        kernel_wins = [3, 4, 5]
    model = TextCNN(tokenizer.vocab_size, dim_channel, kernel_wins, dropout_rate, num_class, embedding_dim, num_class_features)
    return model
