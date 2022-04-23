# NOTE: everything is val for now,
# since the primary goal is to get the model working
# CHANGE THIS FOR ACTUAL USE!

TRAIN_PATH = "./data/train.label.jsonl"
TEST_PATH = "./data/test.label.jsonl"
VALIDATION_PATH = "./data/val.label.jsonl"

VOCAB_PATH = "./data/vocab"
GLOVE_PATH = "./data/glove.6B.50d.txt"

TRAIN_EDGE_PATH = "./data/train.w2s.tfidf.jsonl"
TEST_EDGE_PATH = "./data/test.w2s.tfidf.jsonl"
VAL_EDGE_PATH = "./data/val.w2s.tfidf.jsonl"

SAMPLE_PATH = "./sample/sample.label.jsonl"
SAMPLE_EDGE_PATH = "./sample/sample.w2s.tfidf.jsonl"

# model config
ATTENTION_HEADS = 8
