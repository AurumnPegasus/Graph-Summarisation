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
