# Graph-Summarisation


### Data Preparation

CNNDM (CNN and DailyMail Dataset)
About the dataset - 300k unique news articles written by journalists at CNN and DailyMail

Example - 
{
  "text":["deborah fuller has been banned from keeping animals ... 30mph",...,"a dog breeder and exhibitor... her dogs confiscated"],
  "summary":["warning : ... at a speed of around 30mph",... ,"she was banned from ... and given a curfew "],
  "label":[1,3,6]
}

Json formatted tf-idf dataset


- train.w2s.tfidf.jsonl
- test.w2s.tfidf.jsonl
- val.w2s.tfidf.jsonl
- vocab
- filter_word.txt


### Reproducing Actual Paper Results

Step 1. Download all the code and datasets from here - 
https://drive.google.com/drive/folders/1NjecFgL-vTnBBxhgpvrtTpBft1vhnPi8?usp=sharing

Step 2. Train
- 300 Dimensional embeddings (Actual)
```python HeterSumGraph/train.py --cuda --gpu 0 --data_dir raw/CNNDM --cache_dir cache --embedding_path glove/glove.42B.300d.txt --model HSG --save_root models/ --log_root logs/ --lr_descent --grad_clip -m 3 --word_emb_dim 300```

- 100 Dimensional embeddings (Constrained)
```python HeterSumGraph/train.py --cuda --gpu 0 --data_dir raw/CNNDM --cache_dir cache --embedding_path glove/glove.6B.100d.txt --model HSG --save_root models/ --log_root logs/ --lr_descent --grad_clip -m 3 --word_emb_dim 100```
