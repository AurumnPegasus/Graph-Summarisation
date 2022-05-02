import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.nn.functional import softmax

class HeterSumGraph(nn.Module):
    def __init__(self, dw, ds, dh, de, heads):
        super().__init__()

        self.linear1 = nn.Linear(dw, dh)
        self.linear2 = nn.Linear(ds, dh)
        print("created initial linear layers")


        self.gat1 = GATConv(
            in_channels=(dh, dh), out_channels=dh, heads=heads, edge_dim=de
        )

        self.gat2 = GATConv(
            in_channels=(dh, dh), out_channels=dh, heads=heads, edge_dim=de
        )
        print("created gat layers")

        self.linear3 = nn.Linear(dh, dh)
        self.linear4 = nn.Linear(dh, dh)

        print("created hidden layers")

        self.linear5 = nn.Linear(dh, 1)

        #self.classifierSigmoid = torch.nn.Sigmoid()

    def forward(self, Xw, Xs, E, Erev):
        # passing through the first linear layer
        Hw = self.linear1(Xw)
        Hs = self.linear2(Xs)
        #print(Hw.shape,Hs.shape)

        # passing through the first gat layers
        # for both sentence and word embeddings
        # simultaneously
        nHw = self.gat1((Hs, Hw), E)
        #print(nHw.shape)

        nHs = self.gat2((Hw, Hs), Erev)
        #print(nHs.shape)

        # adding residual connections
        Hw = self.linear4(nHw + Hw)
        Hs = self.linear3(nHs + Hs)

        results = self.linear5(Hs)

        #results = self.classifierSigmoid(results)

        #print(results)
        #r = softmax(results, dim=1)
        #print(r)
        #print(results.shape)
        return(results)

        # returning the final states after the iteration
        #return Hw, Hs


# if __name__ == "__main__":
#     dw = 50
#     ds = 384
#     dh = 64
#     de = 64
#     heads = 1

#     model = HeterSumGraph(dw, ds, dh, de, heads)

#     n = 50
#     m = 500
#     Xw = torch.rand(m, dw)
#     Xs = torch.rand(n, ds)
#     E = torch.randint(0, 50, (2, m))
#     Erev = torch.randint(0, 50, (2, m))
#     Hw, Hs = model.forward(Xw, Xs, E, Erev)

#     print(Hw.shape)
#     print(Hs.shape)
#     assert Hw.shape == (m, dh), "something went wrong for Hw"
#     assert Hs.shape == (n, dh), "something went wrong for Hs"
#     print("success")
