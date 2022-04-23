import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class HeterSumGraph(nn.Module):
    def __init__(self, dw, ds, dh, de, heads):
        super().__init__()

        self.linear1 = nn.Linear(dw, dh)
        self.linear2 = nn.Linear(ds, dh)

        self.gat1 = GATConv(
            in_channels=(dh, dh), out_channels=dh, heads=heads, edge_dim=de
        )

        self.gat2 = GATConv(
            in_channels=(dh, dh), out_channels=dh, heads=heads, edge_dim=de
        )

        self.linear3 = nn.Linear(dh, dh)
        self.linear4 = nn.Linear(dh, dh)

    def forward(self, Xw, Xs, E):
        Hw = self.linear1(Xw)
        Hs = self.linear2(Xs)

        nHw = self.gat2((Hs, Hw), E)  # (5000, 64)
        nHs = self.gat1((Hw, Hs), E)  # (50, 64)

        Hw = self.linear4(nHw + Hw)
        Hs = self.linear3(nHs + Hs)

        return Hw, Hs


def test():
    dw = 50
    ds = 384
    dh = 64
    de = 64
    heads = 1

    model = HeterSumGraph(dw, ds, dh, de, heads)

    n = 50
    m = 5000
    Xw = torch.rand(m, dw)
    Xs = torch.rand(n, ds)
    E = torch.randint(0, 1, (2, m + n))
    Hw, Hs = model.forward(Xw, Xs, E)
    assert Hw.shape == (5000, 64), "something went wrong for Hw"
    assert Hs.shape == (50, 64), "something went wrong for Hs"
    print("success")


test()
