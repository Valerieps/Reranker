import io
# read dev.d100
# count number of queries

dev_d100 = "D:\\Drive\\__TCC__Reranker\\Reranker\\run_msmarco\\data_train\\04-inference_files\\dev.d100.tsv"
dev_queries = set()
with io.open(dev_d100, "r", encoding="utf-8") as f:
    for line in f:
        query_id, query_text, doc_id, url, title, doc_text, something = line.split("\t")
        dev_queries.add(query_id)


# do the same with eval
eval_d100 = "D:\\Drive\\__TCC__Reranker\\Reranker\\run_msmarco\\data_train\\04-inference_files\\eval.d100.tsv"
eval_queries = set()
with open(eval_d100, "r", encoding="utf-8") as f:
    for line in f:
        query_id, query_text, doc_id, url, title, doc_text, something = line.split("\t")
        eval_queries.add(query_id)


print(len(eval_queries))
print(len(dev_queries))
print(eval_queries==dev_queries)

import pickle

"""
Sao identicos no formato, só variam 
5793
5193
False
"""
# chck if dev and eval have no queyy in comom