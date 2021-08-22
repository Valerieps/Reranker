import io
from collections import defaultdict

dev_d100 = "D:\\Drive\\__TCC__Reranker\\Reranker\\run_msmarco\\data_train\\04-inference_files\\dev.d100.tsv"
dev_queries = set()

query_to_annotation = dict()

with io.open(dev_d100, "r", encoding="utf-8") as f:
    for line in f:
        query_id, query_text, doc_id, url, title, doc_text, annotation = line.split("\t")
        dev_queries.add(query_id)
        annotation = annotation.strip()
        if query_id in query_to_annotation:
            query_to_annotation[query_id][annotation] += 1
        else:
            query_to_annotation[query_id] = {'0': 0,
                                             '1' : 0}
            query_to_annotation[query_id][annotation] += 1

print(query_to_annotation)

#
# # do the same with eval
# eval_d100 = "D:\\Drive\\__TCC__Reranker\\Reranker\\run_msmarco\\data_train\\04-inference_files\\eval.d100.tsv"
# eval_queries = set()
# with open(eval_d100, "r", encoding="utf-8") as f:
#     for line in f:
#         query_id, query_text, doc_id, url, title, doc_text, something = line.split("\t")
#         eval_queries.add(query_id)
#
# same_queries = eval_queries.intersection(dev_queries)
# print(len(eval_queries))
# print(len(dev_queries))
# print(eval_queries==dev_queries)
# print(same_queries)

import pickle

"""
Sao identicos no formato, só variam 
5793
5193
False
"""
# chck if dev and eval have no queyy in comom