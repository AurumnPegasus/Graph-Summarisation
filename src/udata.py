from config import SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH
from config import EMBEDDING_SIZE, MASKED

import torch
from torch import utils
from sentence_transformers import SentenceTransformer

import json
import random
import numpy as np
from icecream import ic
from tqdm import tqdm
from collections import defaultdict


class Datas(utils.data.Dataset):
    def __init__(self, data_path, edges_path, embeddings_path):
        # loading the sentence transformer
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        print("loaded sentence transformer")

        # loading the paths for the datasets
        self.data_path = data_path
        self.edges_path = edges_path
        self.embeddings_path = embeddings_path
        print("got paths")

        # initialising masked tokens
        self.masked_init = np.random.rand(MASKED, EMBEDDING_SIZE)
        self.sentword2id = {}
        self.word2idx = defaultdict(int)
        print("initialised masked tokens")

        # getting the contents of these files
        self.data = self.get_contents(self.data_path)
        self.edges = self.get_contents(self.edges_path)
        self.embeddings = self.get_contents(self.embeddings_path)
        print("got data & embeddings")

        # processing embeddings
        self.process_embeddings()
        print("processed embeddings")

    def get_contents(self, path):
        """
        returns the contents of the file at the given path
        """
        with open(path, "r") as f:
            return f.readlines()

    def process_embeddings(self):
        """
        takes each line, splits it by whitespace.
        format: line[0] is the word and line[1:] is the embeddings
        """
        self.embeddings = [line.split() for line in self.embeddings]
        self.embeddings = {
            line[0]: np.array(line[1:], dtype=np.float64)
            for line in tqdm(self.embeddings, "processing embeddings")
        }

    def get_Xw(self, edge):
        """
        embeds the words from the edge data using self.embeddings
        """
        self.word2idx = defaultdict(int)
        Xw = []
        index = 0
        for sentence in edge.values():
            for word in sentence.keys():
                if word not in self.embeddings:
                    word = "unk"

                # Masking token based on probability (MASKED/NUM_WORDS)
                prob = random.uniform(0, 1)
                if prob < 0.02 and len(self.sentword2id) < MASKED:
                    self.sentword2id[(str(index), word)] = len(self.sentword2id)

                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    Xw.append(self.embeddings[word].tolist())
            index += 1

        # Adding the masked tokens at the end
        index = 0
        for key in self.sentword2id.keys():
            Xw.append(self.masked_init[index].tolist())
            self.sentword2id[key] = len(Xw) - 1
            index += 1
        if index < MASKED:
            for m in self.masked_init[index:]:
                Xw.append(m.tolist())

        return torch.FloatTensor(Xw)

    def get_Xs(self, data):
        """
        uses sentence transformer to embed the sentences
        """
        return torch.tensor(self.sentence_transformer.encode(data["text"])).float()

    def get_E(self, data, edge, w_size):
        """
        gets the edge list, a list of (i, j)
        and the reversed edge list of (j, i)
        where i is the sentence number
        and j is the word number
        """

        adjacency = []
        for i, sentence in enumerate(data["text"]):
            current = [0]*w_size
            for word in edge[str(i)]:
                if (str(i), word) not in self.sentword2id:
                    current[self.word2idx[word]] = edge[str(i)][word]
                else:
                    current[self.sentword2id[(str(i), word)]] = edge[str(i)][word]
            adjacency.append(current)
        adj_s2w = torch.FloatTensor(adjacency)
        adj_w2s = torch.transpose(adj_s2w, 0, 1)

        return adj_s2w, adj_w2s

    def get_labels(self, data):
        """
        returns a list of binary values
        1 represents presence in the summary
        0 represents absence
        """
        return torch.tensor([int(i in data["label"]) for i in range(len(data["text"]))])

    def __getitem__(self, i):
        """
        returns one data point, of
        (
            (
                Xw: the word embeddings
                Xs: the sentence embeddings
                E: the edge list
                Erev: the edge list reversed
            ),
            y: the labels
        )
        """
        data = json.loads(self.data[i])
        edge = json.loads(self.edges[i])
        Xw = self.get_Xw(edge)
        Xs = self.get_Xs(data)
        E, Erev = self.get_E(data, edge, Xw.shape[0])
        y = self.get_labels(data)
        ic(Xw.shape, Xs.shape, E.shape, Erev.shape)

        return (Xw, Xs, E, Erev), y

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    datas = Datas(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
    dataloader = utils.data.DataLoader(datas)
    c = 0
    for source, target in tqdm(dataloader):
        Xw, Xs, E, Erev = source
        ic(Xw.shape, Xs.shape, E.shape, Erev.shape, target.shape)
        c += 1
        if c == 10:
            break
    print("success")
