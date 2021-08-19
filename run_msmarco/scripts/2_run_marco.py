# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from reranker import Reranker, RerankerDC
from reranker import RerankerTrainer, RerankerDCTrainer
from reranker.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

# logger = logging.getLogger(__name__)


"""
--nproc_per_node 4 examples/msmarco-doc/2_run_marco.py \
--output_dir model_checkpoints \
--model_name_or_path  bert-base-uncased \
--do_train \
--save_steps 2000 \
--train_dir data/mini-data/ \
--max_len 512 \
--fp16 \
--per_device_train_batch_size 1 \
--train_group_size 8 \
--gradient_accumulation_steps 1 \
--per_device_eval_batch_size 64 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--overwrite_output_dir \
--dataloader_num_workers 8 \
"""


def main():
    print("\n\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    # setup_logging(data_args, model_args, training_args)
    # print()

    # Set seed
    set_seed(training_args.seed)

    # 1? Não seria 2?
    num_labels = 1

    print("\n=========== CONFIGURANDO AS COISAS ===========")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,  # bert-base-uncased
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    _model_class = RerankerDC if training_args.distance_cache else Reranker

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # data_args.train_path é uma lista com o nome de todos os arquivos TSV ou JSON
    # dentro do train_dir.

    # Get datasets #true
    print("\n===========  GETTING DATASET ===========")
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            args=data_args,
            path_to_tsv=data_args.train_path,  # é uma lista
            tokenizer=tokenizer,
            train_args=training_args)
    else:
        train_dataset = None

    # Initialize our Trainer
    print("\n===========  INITIALIZING TRAINER ===========")
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
    )

    # ate aqui foi OK

    # Training

    if training_args.do_train:
        print("\n===========  TRAINING ===========")
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        print("\n===========  EVALUATING ===========")
        trainer.evaluate()

    if training_args.do_predict:
        print("\n===========  PREDICTION ===========")
        logging.info("*** Prediction ***")

        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = PredictionDataset(
            data_args.pred_path, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )
        assert data_args.pred_id_file is not None

        pred_qids = []
        pred_pids = []
        with open(data_args.pred_id_file) as f:
            for l in f:
                q, p = l.split()
                pred_qids.append(q)
                pred_pids.append(p)

        pred_scores = trainer.predict(test_dataset=test_dataset).predictions

        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            with open(data_args.rank_score_path, "w") as writer:
                for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                    writer.write(f'{qid}\t{pid}\t{score}\n')
    print("\n===========  TERMINOU ===========")


def setup_logging(data_args, model_args, training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()