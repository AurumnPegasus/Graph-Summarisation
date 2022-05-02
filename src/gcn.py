import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.W = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W)

        self.b = nn.Parameter(torch.zeros(out_channels))

    def forward(self, X, A):
        updated_features = torch.mm(X, self.W)
        edge_removal = torch.mm(A, updated_features)
        return edge_removal + self.b
