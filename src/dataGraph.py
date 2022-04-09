import pandas as pd
from icecream import ic
from constants import *
import numpy as np
from dataSplit import Data
from sentence_transformers import SentenceTransformer
import math

class Graph:
    def __init__(self,
                 dataset):
        self.dataset = dataset

        self.w2id = {}
        self.id2w = {}
        self.w2emb = {}

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def prepEmbeddings(self):
        with open('./glove.6B.50d.txt', 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.w2emb[word] = embedding

    def getGraph(self, sentences, edges):

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
        ic(embeddings.shape, word_embeddings.shape, adjacency.shape)

    def handle(self):
        df = pd.read_json(path_or_buf='./data/cache/CNNDM/val.w2s.tfidf.jsonl', lines=True)
        self.prepEmbeddings()

        for i in range(len(self.dataset)):
            sentences = self.dataset.iloc[i]
            edges = df.iloc[i]
            self.getGraph(sentences, edges)
            break

if __name__ == "__main__":
    d = Data()
    d.handle()
    train, test, val = d.getDFs()
    g = Graph(val)
    g.handle()
