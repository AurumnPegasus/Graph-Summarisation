from udata import Datas
from umodel import HeterSumGraph

from config import SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH
from config import EMBEDDING_PATH, MASKED

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from icecream import ic
from tqdm import tqdm

if __name__ == "__main__":
    dataset = Datas(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
    dataloader = utils.data.DataLoader(dataset, batch_size=1)
    for x, y in tqdm(dataloader):
        ic(x, y)
        break


