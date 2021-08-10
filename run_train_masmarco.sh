#!/bin/bash

#python -m torch.distributed.launch \
python -m torch.distributed.launch \
  --nproc_per_node 1 examples/msmarco-doc/run_marco.py \
  --output_dir checkpoints \
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