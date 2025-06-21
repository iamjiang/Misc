#!/bin/bash

path='/workspace/TR-Project'

CUDA_VISIBLE_DEVICES=0 python $path/train.py \
--model_path answerdotai/ModernBERT-base   \
--train_batch 4 \
--eval_batch 8 \
--gradient_accumulation_step 12 \
--num_epochs 5 \
--max_context_length 8192



