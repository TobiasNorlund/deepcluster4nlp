import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, vocab_size, padding_token_id, dim_channel, kernel_wins, dropout_rate, num_class, embedding_dim, num_class_features=4096):
        super(TextCNN, self).__init__()

        self.padding_token_id = padding_token_id
        self.kernel_wins = kernel_wins
        self.num_class_features = num_class_features
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, dim_channel, w) for w in kernel_wins])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class_features)

        self.relu = nn.ReLU(inplace=True)
        self.top_layer = nn.Linear(num_class_features, num_class)

    def forward(self, x, **kwargs):

        emb_x = self.embed(x)
        # emb_x = emb_x.unsqueeze(1)

        emb_x = emb_x.permute(0, 2, 1)
        con_x = [conv(emb_x) for conv in self.convs]

        # remove convolution influence from padding tokens
        # we do this by greatly reducing the conv results influenced by padding tokens before the pooling
        for i, con_tensor in enumerate(con_x):
            filter_size = self.kernel_wins[i]
            m = ((x == self.padding_token_id)*(-100)).unsqueeze(1)
            m = m[:, :, filter_size-1:]
            con_x[i] = con_tensor + m

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)

        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)

        fc_x = self.fc(fc_x)

        # print(f"fc: {fc_x}")

        if self.top_layer:
            fc_x = self.relu(fc_x)
            # print(f"Relu: {fc_x}")
            fc_x = self.top_layer(fc_x)

        return fc_x

    def set_top_layer(self, cluster_list_length):
        # set last fully connected layer
        # cluster_list_length = len(deepcluster.cluster_lists)

        self.top_layer = nn.Linear(self.num_class_features, cluster_list_length)
        self.top_layer.weight.data.normal_(0, 0.1)
        self.top_layer.bias.data.zero_()
        # self.top_layer.cuda()
    
    def reset_top_layer(self):
        self.top_layer = None


def textcnn(tokenizer, dim_channel=100, kernel_wins=None, dropout_rate=0.5, num_class=2, embedding_dim=300, num_class_features=4096):
    if kernel_wins is None:
        kernel_wins = [3, 4, 5]

    padding_token_id = tokenizer.pad_token_id
    model = TextCNN(tokenizer.vocab_size, padding_token_id, dim_channel, kernel_wins, dropout_rate, num_class, embedding_dim, num_class_features)
    return model
