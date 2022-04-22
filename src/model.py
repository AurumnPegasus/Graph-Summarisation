from torch_geometric.nn import GATConv
import torch.nn as nn
from torch.nn import Linear
import torch
import numpy as np


class HeterSumGraph(nn.Module):
    def __init__(self, Xw, Xs, E, heads):
        """
        Xw: (num_words, word_embed_size)
        Xs: (num_sentences, sent_embed_size)
        E: (num_sentences, num_words)

        for validation's 0th sample these are
        (29, 384) and (29, 253) and (29, 253)
        respectively

        NOTE: does not handle edges yet!
        to be updated in the next version

        NOTE: only does 1 layer of message passing,
        as mentioned in the paper.
        """
        super().__init__()

        Xw = torch.from_numpy(Xw).to(torch.float)
        Xs = torch.from_numpy(Xs).to(torch.float)
        E = torch.from_numpy(E).to(torch.long)

        self.gat_sent2word = GATConv(
            in_channels=Xs.shape[1],
            out_channels=Xw.shape[1],
            heads=heads,
        )

        self.gat_word2sent = GATConv(
            in_channels=Xw.shape[1],
            out_channels=Xs.shape[1],
            heads=heads,
        )

        self.linear_word = Linear(in_features=Xw.shape[1], out_features=Xw.shape[1])
        self.linear_sent = Linear(in_features=Xs.shape[1], out_features=Xs.shape[1])

        self.hidden_word = Xw
        self.hidden_sent = Xs

        self.E = torch.transpose(E, 0, 1)
        self.Erev = []
        for item in E:
            self.Erev.append((item[1], item[0]))
        self.Erev = torch.transpose(torch.tensor(self.Erev), 0, 1)

    def forward(self):
        # passing word and sentence through the GAT layers simultaneously
        temp_word_next = self.gat_sent2word(self.hidden_sent, edge_index=self.E)
        temp_sent_next = self.gat_word2sent(self.hidden_word, edge_index=self.Erev)

        # adding residual connections
        hidden_word_next = self.linear_word(temp_word_next + self.hidden_word)
        hidden_sent_next = self.linear_word(temp_sent_next + self.hidden_sent)

        return hidden_word_next, hidden_sent_next
