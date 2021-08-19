# Data Pipeline

step                   | input                              | script              |  output
-----------------------|------------------------------------|---------------------|-----------------------
**PREP FOR TRAIN**     | hdct-marco-train<br>msmarco-doctrain-qrels.tsv<br>msmarco-doctrain-queries.tsv<br>msmarco-docs.tsv | 1_preprocess.py | preprocessed_files_for_training
**TRAIN**              | preprocessed_files_for_training    | 2_run_marco.py         | chekpoints
**PREP FOR INFERENCE** | dev.d100.tsv                       | 3_topk_text_2_json.py    | all.json<br>ids.tsv
**INFERENCE**          | checkpoints<br>all.json<br>ids.tsv | 2_run_marco.py         | score.txt 
**MARCO SCORING**      | score.txt                          | 4_score_to_marco.py      | score.txt.marco
**MAKE REFERENCE FILE**| ======= ?? =======                 | 5_make_reference_file.py | reference_file.tsv  
**CALCULATE MRR**      |reference_file.tsv<br>score.txt.marco| 6_msmarco_eval.py    | **MRR on screen** 

# Files Details

file | description | schema
---|---|---
hdct-marco-train | 100 best docs for each query | [queryID, docID, rank]
msmarco-doctrain-queries.tsv | complete query text | [queryID, query_text]
msmarco-doctrain-qrels.tsv | Conects a query_id to a relevant doc_id | [queryID, iteration, docID, relevancy]
msmarco-docs.tsv | Docs to be ranked | [docID, url, title, docText]
training_files | tokenized query and docs, with positive and negative examples |  see sample
dev.d100.tsv | | [queryID, query, docID, url, title, docText] 
all.json | 
ids.tsv |
score.txt |
score.txt.marco | 
reference_file.tsv | 

# Scripts Details

### 1_preprocess.py





HDCT Files
---------------------------------
max de 366 mi de linhas
(ranking dos 100 melhores doc para cada query)

[topic_id, doc_id, rank]

Ex.:
1185869	D59221	1
1185869	D59220	2
...
1185869	D4570	1000

* Vai sempre ate mil.
* Para cada palavra, seleciona-se até o doc 1000

* pq dividido em muitos txt?


MSMARCO DOCS
----------------------------------
3.213.835 linhas
(dados dos documentos)

[docID, url, title, docText]

Ex.:
D1555982	https://answers.yahoo.co...114826AAwCFvR	The hot glowing....electromagnetic radiation.?	Science & Mathematics Physics ... s


DEV.D100.TSV
---------------------------------
519.300 linhas

[query doc seq]
[queryID?, query(?) docID url title docText]

* as queries desse doc não tem nenhuma sobreposição com o atquivo msmarco-queries
* falta descobrir como isso é usado no Reranker
* De onde vem essas queries?


msmarco-doctrain-qrels.tsv
------------------------------
8.680 linhas

[queryID, iteration, docID, relevancy]

"queryID": In qrels format the field name is TOPIC
"iteration": is the feedback iteration (almost always zero and not used),
"docID": the official document number that corresponds to the "docno" field in the documents, and
"relevancy": 0 for not relevant and 1 for relevant.

Ex.:
3 0 D312959 1

* 319.928 unique docs have qrels