# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas
import gzip
import csv
import json
import os
import datasets
import random

from argparse import ArgumentParser
from transformers import AutoTokenizer
from collections import defaultdict
from tqdm import tqdm


def main():
    # cache_dir="/content/drive/My Drive/_RERANKER_/data"
    # cache_dir = "/cache"
    args = read_args()

    topic_to_relevant_documents = read_qrel(args)
    print(f"{len(topic_to_relevant_documents)} topics found")

    high_hdct_rank_no_qrel = get_queries_with_relevance_judgment(args, topic_to_relevant_documents)
    print(f"Topics that have no qrel in MSMarco, but were ranked high (top 1000) by HDCT: {len(high_hdct_rank_no_qrel)}")

    # Get docs and queries
    print("Montando datasets do formato Arrow")
    columns = ['doc_id', 'url', 'title', 'body']
    train_doc_collection = datasets.load_dataset(
        path='csv',
        data_files=args.doc_collection,
        column_names=columns,
        delimiter='\t',
        ignore_verifications=True,
        # cache_dir=cache_dir
    )['train']

    train_query_collection = datasets.load_dataset(
        path='csv',
        data_files=args.query_collection,
        column_names=['query_id', 'qry'],
        delimiter='\t',
        ignore_verifications=True,
        # cache_dir=cache_dir
    )['train']

    """
    O objeto Dataset guarda cada linha como um dict
    
    print(train_doc_collection[0])> 
    {'doc_id': 'D1555982', 
     'url': 'https://answers.yahoo.com/question/index?qid=20071007114826AAwCFvR',
     'title': 'The hot glowing surfaces of stars emit energy in the form of electromagnetic radiation.?',
     'body': 'Science & Math...rder contacts online? '
     }
    """

    print("Mapping docs and queries to index")
    doc_map = {x['doc_id']: idx for idx, x in enumerate(train_doc_collection)} # maps the doc_id to its position in Dataset object
    qry_map = {str(x['query_id']): idx for idx, x in enumerate(train_query_collection)} # maps the query_id to its position in Dataset object

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    out_file = args.rank_file # 000.txt

    # pega o stem do nome e monta o nome do output
    if out_file.endswith('.tsv') or out_file.endswith('.txt'):
        out_file = out_file[:-4]
    out_file = os.path.join(args.json_dir, os.path.split(out_file)[1])
    out_file = out_file + '.group.json'

    # pq essas queries sao importantes? Que nao tem qrels?
    queries = list(high_hdct_rank_no_qrel.keys())
    print("Fazendo alguma coisa que eu não sei o que é")
    with open(out_file, 'w') as f:
        for qid in tqdm(queries):

            # pick from top of the full initial ranking
            # negs é o que, negatives?
            negs = high_hdct_rank_no_qrel[qid][:args.sample_from_top]  # sample_from_top atualmente = 100

            # shuffle if random flag is on
            if args.random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:args.n_sample] # n_sample atualmente é 10

            # Encodings
            neg_encoded = encode_passages(args, columns, doc_map, neg_encoded, negs, tokenizer, train_doc_collection)
            pos_encoded = pos_encoding(args, columns, doc_map, qid, tokenizer, train_doc_collection)
            query_dict = encode_query(args, qid, qry_map, tokenizer, train_query_collection)

            item_set = {
                'qry': query_dict,
                'pos': pos_encoded,
                'neg': neg_encoded,
            }

            f.write(json.dumps(item_set) + '\n')

def read_args():
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', required=True)              # bert-base-uncased
    parser.add_argument('--rank_file', required=True)                   # hdct-marco-train txt files
    parser.add_argument('--truncate', type=int, default=512)

    parser.add_argument('--sample_from_top', type=int, required=True)   # 100
    parser.add_argument('--n_sample', type=int, default=100)            # 10
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--json_dir', required=True)                    # data/training_file

    parser.add_argument('--qrel', required=True)                        # msmarco-doctrain-qrels.tsv
    parser.add_argument('--query_collection', required=True)            # msmarco-doctrain-queries.tsv
    parser.add_argument('--doc_collection', required=True)              # msmarco-docs.tsv
    args = parser.parse_args()
    return args


def read_qrel(args):
    """ Mapeia topicid para lista de docID relevantes
    cada topico pode estar relacionado com mais de um doc relevante para ele
    topic é a query

    Return: topic_to_relevant_documents (dict)

    """

    topic_to_relevant_documents = {}
    with gzip.open(args.qrel, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicID, _, docid, relevancy] in tsvreader:
            assert relevancy == "1"
            if topicID in topic_to_relevant_documents:
                topic_to_relevant_documents[topicID].append(docid)
            else:
                topic_to_relevant_documents[topicID] = [docid]
    return topic_to_relevant_documents


def get_queries_with_relevance_judgment(args, topic_to_relevant_documents):
    """

    line format [topic_id, doc_id, ranking]
    Return:
        high_hdct_rank_no_qrel: every topic that has no qrel in MSMarco, but was ranked high (top 1000) by HDCT

    """
    high_hdct_rank_no_qrel = defaultdict(list)  # defaultdict never raises a KeyError. If key not present, adds the default value
    no_relevance_judgement = set()
    with open(args.rank_file) as f:  # hdct-marco-train
        for line in f:
            topic_id, doc_id, rank = line.split()

            # se o topic do hdct nao estiver no conjunto de qrels do msmarco
            if topic_id not in topic_to_relevant_documents:
                no_relevance_judgement.add(topic_id)
                continue

            # Se doc é relevante para aquela query, de acordo com a msmarco
            # ou seja, se as duas anotacoes concordam
            # segue a vida
            if doc_id in topic_to_relevant_documents[topic_id]:
                continue


            # append passage if & only if it is not judged relevant by MSmarco anotation, but ranks high by HDCT
            high_hdct_rank_no_qrel[topic_id].append(doc_id)
    print(f'{len(no_relevance_judgement)} queries not judged and skipped', flush=True)
    return high_hdct_rank_no_qrel


def encode_query(args, qid, qry_map, tokenizer, train_query_collection):
    q_idx = qry_map[qid]
    query_dict = {
        'qid': qid,
        'query': tokenizer.encode(
            train_query_collection[q_idx]['qry'],
            add_special_tokens=False,
            max_length=args.truncate,
            truncation=True),
    }
    return query_dict


def pos_encoding(args, columns, doc_map, qid, tokenizer, train_doc_collection):
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
    return pos_encoded


def encode_passages(args, columns, doc_map, neg_encoded, negs, tokenizer, train_doc_collection):
    neg_encoded = []
    for neg_id in negs:
        idx = doc_map[neg_id]
        item = train_doc_collection[idx]  # item segura o dict do doc com dados e metadados
        did, url, title, body = (item[k] for k in columns)
        url, title, body = map(lambda v: v if v else '', [url, title, body])  # Elimina None, mapeia pra empty string

        # documentar o encoding padrao que usa url, titulo, body
        encoded_neg = tokenizer.encode(
            url + tokenizer.sep_token + title + tokenizer.sep_token + body,
            add_special_tokens=False,
            max_length=args.truncate,
            truncation=True
        )
        neg_encoded.append({
            'passage': encoded_neg,
            'pid': neg_id,
        })
    return neg_encoded


if __name__ == "__main__":
    main()
