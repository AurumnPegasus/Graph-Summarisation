# Graph-Summarisation


### Data Preparation

CNNDM (CNN and DailyMail Dataset)
About the dataset - 300k unique news articles written by journalists at CNN and DailyMail

Example - 
{'id': '0054d6d30dbcad772e20b22771153a2a9cbeaf62',
 'article': '(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour.'
 'highlights': 'The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says .'}

Json formatted tf-idf dataset


- train.w2s.tfidf.jsonl
- test.w2s.tfidf.jsonl
- val.w2s.tfidf.jsonl
- vocab
- filter_word.txt
