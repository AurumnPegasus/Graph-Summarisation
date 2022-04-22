# loading required stuff
from data_split import Data
from data_graph import Graph
from constants import VALIDATION_PATH, VAL_EDGE_PATH, ATTENTION_HEADS
from model import HeterSumGraph

# loading utils
import pandas as pd

data = Data(VALIDATION_PATH)
df1 = data.get_df()
print("read df1")
df2 = pd.read_json(path_or_buf=VAL_EDGE_PATH, lines=True)
print("read df2")
data_loader = Graph(df1, df2)
print("data loader setup done")


# count = 0
# for x, y in data_loader:
#     embeddings, edge_list, word_embeddings = x
#     print(embeddings.shape)
#     print(edge_list.shape)
#     print(word_embeddings.shape)
#     print(y.shape)
#     print()

#     """
#     PRINTS OUT:
#     (29, 384)
#     (29, 253)
#     (253, 50)
#     (29,)

#     (17, 384)
#     (17, 115)
#     (115, 50)
#     (17,)
#     """

#     if count == 1:
#         break
#     count += 1

for x, y in data_loader:
    embeddings, edge_list, word_embeddings = x
    graph = HeterSumGraph(embeddings, word_embeddings, edge_list, heads=ATTENTION_HEADS)
    print(graph)
    word_next, sent_next = graph.forward()
    print(word_next.shape, sent_next.shape)
    break
