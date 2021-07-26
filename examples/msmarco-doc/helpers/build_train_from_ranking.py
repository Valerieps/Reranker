# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
from collections import defaultdict
import datasets
import random
from tqdm import tqdm
import pandas

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--rank_file', required=True) # hdct-marco-train file
parser.add_argument('--truncate', type=int, default=512)

parser.add_argument('--sample_from_top', type=int, required=True)
parser.add_argument('--n_sample', type=int, default=100)
parser.add_argument('--random', action='store_true')
parser.add_argument('--json_dir', required=True)

parser.add_argument('--qrel', required=True)
parser.add_argument('--query_collection', required=True)
parser.add_argument('--doc_collection', required=True)
args = parser.parse_args()


def read_qrel():
    """ Cria um dict que mapeia topicid: docid
    cada topico pode estar relacionado com mais de um doc relevante para ele
    topic é tipo a query
    """
    import gzip, csv
    qrel = {}
    with gzip.open(args.qrel, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


def get_queries_with_qrels(qrels_to_relevant_docs):
    rankings = defaultdict(list)  # defaultdict never raises a KeyError. If key nor present, adds the default value
    no_relevance_judgement = set()
    with open(args.rank_file) as f:  # hdct-marco-train
        for line in f:
            qid, pid, rank = line.split()
            if qid not in qrels_to_relevant_docs:
                no_relevance_judgement.add(qid)
                continue
            if pid in qrels_to_relevant_docs[qid]:
                continue
            # append passage if & only if it is not judged relevant but ranks high
            rankings[qid].append(pid)
    print(f'{len(no_relevance_judgement)} queries not judged and skipped', flush=True)
    return rankings


def main():
    topics_to_relevant_docs = read_qrel()
    print(f"{len(topics_to_relevant_docs)} topics found")
    rankings = get_queries_with_qrels(topics_to_relevant_docs)
    print(len(rankings))

    columns = ['did', 'url', 'title', 'body']
    collection_path = args.doc_collection
    train_doc_collection = datasets.load_dataset(
        path='csv',
        data_files=collection_path,
        column_names=columns,
        delimiter='\t',
        ignore_verifications=True,
    )['train']

    train_query_collection_path = args.query_collection
    train_query_collection = datasets.load_dataset(
        'csv',
        data_files=train_query_collection_path,
        column_names=['qid', 'qry'],
        delimiter='\t',
        ignore_verifications=True,
    )['train']

    doc_map = {x['did']: idx for idx, x in enumerate(train_doc_collection)}
    qry_map = {str(x['qid']): idx for idx, x in enumerate(train_query_collection)}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    out_file = args.rank_file
    if out_file.endswith('.tsv') or out_file.endswith('.txt'):
        out_file = out_file[:-4]
    out_file = os.path.join(args.json_dir, os.path.split(out_file)[1])
    out_file = out_file + '.group.json'

    queries = list(rankings.keys())
    with open(out_file, 'w') as f:
        for qid in tqdm(queries):
            # pick from top of the full initial ranking
            negs = rankings[qid][:args.sample_from_top]
            # shuffle if random flag is on
            if args.random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:args.n_sample]

            neg_encoded = []
            for neg in negs:
                idx = doc_map[neg]
                item = train_doc_collection[idx]
                did, url, title, body = (item[k] for k in columns)
                url, title, body = map(lambda v: v if v else '', [url, title, body])
                encoded_neg = tokenizer.encode(
                    url + tokenizer.sep_token + title + tokenizer.sep_token + body,
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                neg_encoded.append({
                    'passage': encoded_neg,
                    'pid': neg,
                })
            pos_encoded = []
            for pos in topics_to_relevant_docs[qid]:
                idx = doc_map[pos]
                item = train_doc_collection[idx]
                did, url, title, body = (item[k] for k in columns)
                url, title, body = map(lambda v: v if v else '', [url, title, body])
                encoded_pos = tokenizer.encode(
                    url + tokenizer.sep_token + title + tokenizer.sep_token + body,
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                pos_encoded.append({
                    'passage': encoded_pos,
                    'pid': pos,
                })
            q_idx = qry_map[qid]
            query_dict = {
                'qid': qid,
                'query': tokenizer.encode(
                    train_query_collection[q_idx]['qry'],
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True),
            }
            item_set = {
                'qry': query_dict,
                'pos': pos_encoded,
                'neg': neg_encoded,
            }
            f.write(json.dumps(item_set) + '\n')

if __name__ == "__main__":
    main()
