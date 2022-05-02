from config import EMBEDDING_PATH, SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH

from data import DataLoader
from model import HeterSumGraph

dw = 50
ds = 384
dh = 64
de = 64
heads = 1

data_loader = DataLoader(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
print("created data loader")

model = HeterSumGraph(dw, ds, dh, de, heads)
print("created model")

for (Xw, Xs, E, Erev), y in data_loader:
    print(Xw.shape,Xs.shape,E.shape,Erev.shape,y)

    Hw, Hs = model.forward(Xw, Xs, E, Erev)
    print(Hw.shape)
    print(Hs.shape)

    # do the needful with the output
    # may need to add more layers to the model itself
