import torch
from gcn import GCNLayer
import torch.nn as nn
from icecream import ic

class HeterSumGraph(nn.Module):
    def __init__(self, dw, ds, dh):
        super().__init__()

        self.linear1 = nn.Linear(dw, dh)
        self.linear2 = nn.Linear(ds, dh)
        print("created initial linear layers")


        self.gcn1 = GCNLayer(in_channels=dh, out_channels=dh)
        self.gcn2 = GCNLayer(in_channels=dh, out_channels=dh)
        print("created gat layers")

        self.linear3 = nn.Linear(dh, ds)
        self.linear4 = nn.Linear(dh, dw)

        print("created hidden layers")

        self.ls1 = nn.LogSoftmax(dim=1)
        self.ls2 = nn.LogSoftmax(dim=1)

    def forward(self, Xw, Xs, E, Erev):

        Hw = self.linear1(Xw)
        Hs = self.linear2(Xs)

        nHw = self.gcn1(Hs, Erev)
        nHs = self.gcn2(Hw, E)

        Hw = self.linear4(nHw + Hw)
        Hs = self.linear3(nHs + Hs)

        Hw, Hs = self.ls1(Hw), self.ls2(Hs)
        return Hw, Hs


if __name__ == "__main__":
    dw = 50
    ds = 384
    dh = 64
    de = 64
    heads = 1

    model = HeterSumGraph(dw, ds, dh)

    n = 50
    m = 500
    Xw = torch.rand(m, dw)
    Xs = torch.rand(n, ds)
    E = torch.randint(0, 50, (2, m))
    Erev = torch.randint(0, 50, (2, m))
    Hw, Hs = model.forward(Xw, Xs, E, Erev)

    print(Hw.shape)
    print(Hs.shape)
    assert Hw.shape == (m, dh), "something went wrong for Hw"
    assert Hs.shape == (n, dh), "something went wrong for Hs"
    print("success")
