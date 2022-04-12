import pandas as pd
from icecream import ic
from constants import TRAIN_EDGE_PATH, TEST_EDGE_PATH, VAL_EDGE_PATH
import numpy as np
from dataSplit import Data
from sentence_transformers import SentenceTransformer
import torch
import torch.utils as utils

class Graph(utils.data.Dataset):
    def __init__(self,
                 dataset,
                 df):
        self.dataset = dataset
        self.df = df

        self.w2emb = {}

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        with open('./glove.6B.50d.txt', 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.w2emb[word] = embedding

    def __len__(self):
        return len(self.dataset)

    def prepEmbeddings(self):
        with open('./glove.6B.50d.txt', 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.w2emb[word] = embedding

    def __getitem__(self, index):

        sentences = self.dataset.iloc[index]
        edges = self.df.iloc[index]

        labels = sentences['label']
        sentences = sentences['text']
        embeddings = self.encoder.encode(sentences)

        vocab = set()
        for edge in edges:
            if isinstance(edge, dict):
                for word in edge:
                    vocab.add(word.lower())

        word_embeddings = []
        for word in vocab:
            if word in self.w2emb:
                word_embeddings.append(self.w2emb[word])
            else:
                word_embeddings.append(self.w2emb["unk"])
        word_embeddings = np.array(word_embeddings)

        adjacency = []
        for i in range(len(sentences)):
            current = []
            for word in vocab:
                if word in edges[i]:
                    current.append(edges[i][word])
                else:
                    current.append(0)
            adjacency.append(current)
        adjacency = np.array(adjacency)

        actual_labels = [0]*len(sentences)
        for label in labels:
            actual_labels[label] = 1

        return (embeddings, adjacency, word_embeddings), actual_labels

if __name__ == "__main__":
    d = Data()
    d.handle()
    train, test, val = d.getDFs()
    val_df = pd.read_json(path_or_buf=VAL_EDGE_PATH, lines=True)
    val_loader = Graph(val, val_df)

