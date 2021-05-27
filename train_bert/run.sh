#!/bin/bash


python3 -u run_pretraining.py \
  --input_file=/yourpath/data/bert/ende.word.train.tfrecord \
  --output_dir=/yourpath/data/bert/pretraining_output_word \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=/yourpath/data/bert/bert_config_word.json \
  --max_seq_length=200 \
  --max_predictions_per_seq=30 \
  --learning_rate=1e-4 \
  --train_batch_size=16 \
  > log_word 2>&1
