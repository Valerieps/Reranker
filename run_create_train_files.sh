#!/bin/bash

#export HF_DATASETS_CACHE="/content/drive/My Drive/_RERANKER_/data"
#export HF_DATASETS_CACHE="/cache"

python examples/msmarco-doc/helpers/build_train_from_ranking.py \
    --tokenizer_name bert-base-uncased \
    --rank_file 	data/hdct-marco-train/000.txt \
    --json_dir data/training_file \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel data/msmarco-doctrain-qrels.tsv.gz \
    --query_collection data/msmarco-doctrain-queries.tsv \
    --doc_collection data/docs_that_have_qrel.tsv


#for i in $(seq -f "%03g" 0 183)
#do
#python helpers/build_train_from_ranking.py \
#    --tokenizer_name bert-base-uncased \
#    --rank_file 	data/hdct-marco-train\${i}.txt \
#    --json_dir data/training_file \
#    --n_sample 10 \
#    --sample_from_top 100 \
#    --random \
#    --truncate 512 \
#    --qrel data/msmarco-doctrain-qrels.tsv.gz \
#    --query_collection data/msmarco-doctrain-queries.tsv \
#    --doc_collection data/msmarco-docs.tsv
#done