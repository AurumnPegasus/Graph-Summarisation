from udata import Datas
from umodel import HeterSumGraph

from config import SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH
from config import EMBEDDING_PATH, MASKED
from config import WORD_DIMENSION, SENTENCE_DIMENSION, HIDDEN_DIMENSION
from config import LR

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.functional as F

from icecream import ic
from tqdm import tqdm

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = Datas(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
    dataloader = utils.data.DataLoader(dataset)
    model = HeterSumGraph(WORD_DIMENSION, SENTENCE_DIMENSION, HIDDEN_DIMENSION)
    loss = nn.KLDivLoss(reduction="mean", log_target=True)
    losses = []
    itera = []
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for e in tqdm(range(10)):
        c = 0
        av = 0
        for x, y in dataloader:
           Xw, Xs, E, Erev = x
           y_index, y_value = y
           Xw, Xs, E, Erev, y_index, y_value = Xw.squeeze(), Xs.squeeze(), E.squeeze(), Erev.squeeze(), y_index.squeeze(),y_value.squeeze()
           Xw, Xs = model.forward(Xw, Xs, E, Erev)
           train_x = torch.index_select(Xw, 0, y_index)
           # y_value = F.log_softmax(y_value, dim=1)
           l = loss(train_x, y_value)
           av += l.item()
           c += 1
           l.backward()
           optimizer.step()
        itera.append(e)
        losses.append(av/c)

    plt.plot(itera, losses)
    plt.show()
