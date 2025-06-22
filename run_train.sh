#!/bin/bash

path='/mnt/d/TR-Project'

CUDA_VISIBLE_DEVICES=0 python $path/train.py \
--model_path answerdotai/ModernBERT-base   \
--train_batch 2 \
--eval_batch 3 \
--gradient_accumulation_step 24 \
--num_epochs 5 \
--output_dir model_output_512 \
--max_context_length 512



