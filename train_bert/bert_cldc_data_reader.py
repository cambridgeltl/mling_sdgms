# -*- coding: UTF-8 -*-
# !/usr/bin/python3
"""
Bert CLDC Data Reader
"""

# ************************************************************
# Imported Libraries
# ************************************************************
import os

from tqdm import tqdm
from collections import Counter
import random

import numpy as np
import torch

import pdb


from data_model.classification_base_data_reader import CLSBaseDataReader

class BERTCLDCDataReader(CLSBaseDataReader):
  def __init__(self, params, tokenizer):
    # cldc data path
    train_lang_idx = 0 if params.cldc_lang[0] == 'en' else 1
    self.train_cldc_path = params.cldc_path[train_lang_idx]
    assert (os.path.exists(self.train_cldc_path))
    texts = torch.load(self.train_cldc_path)
    #train_texts, dev_texts, _ = texts['train'], texts['dev'], texts['test']
    train_texts, dev_texts, _ = texts['train.1000'], texts['dev'], texts['test']

    test_lang_idx = 0 if params.cldc_lang[1] == 'en' else 1
    self.test_cldc_path = params.cldc_path[test_lang_idx]
    assert (os.path.exists(self.test_cldc_path))
    texts = torch.load(self.test_cldc_path)
    #_, _, test_texts = texts['train'], texts['dev'], texts['test']
    tgt_train_texts = None
    if test_lang_idx == train_lang_idx:
      _, _, test_texts = texts['train.1000'], texts['dev'], texts['test']
    else:
      tgt_train_texts, _, test_texts = texts['train.1000'], texts['dev'], texts['test']

    for l in params.cldc_label2idx:
      train_texts[l] = [d.strip().split('\n') for d in train_texts[l]]
      dev_texts[l] = [d.strip().split('\n') for d in dev_texts[l]]
      test_texts[l] = [d.strip().split('\n') for d in test_texts[l]]

    # cldc labels
    self.label2idx = params.cldc_label2idx
    self.idx2label = params.cldc_idx2label
    self.label_size = 4

    # read train
    # different scale for crosslingual training data
    scale = params.scale

    # get the max of sentence length, [CLS], [SEP]
    self.max_text_len = 198

    self.train_idxs, self.train_lens, self.max_train_len, \
    self.rest_train_idxs, self.rest_train_lens, self.train_prop = self.get_data(params, train_texts,
                                                                                tokenizer,
                                                                                train=True,
                                                                                scale=scale)
    # for zero-shot
    if tgt_train_texts is not None:
      _, _, _, \
      self.rest_train_idxs, self.rest_train_lens, _ = self.get_data(params, tgt_train_texts,
                                                                             tokenizer,
                                                                             train=True,
                                                                             scale=.0)

    # read dev data
    self.dev_idxs, self.dev_lens, self.max_dev_len, self.dev_prop = self.get_data(params, dev_texts,
                                                                                  tokenizer)
    # read test data
    self.test_idxs, self.test_lens, self.max_test_len, self.test_prop = self.get_data(params, test_texts,
                                                                                      tokenizer)

    for i in range(self.label_size):
      self.train_idxs[i] = self.pad_texts(self.train_idxs[i], self.max_text_len, tokenizer)
      self.dev_idxs[i] = self.pad_texts(self.dev_idxs[i], self.max_text_len, tokenizer)
      self.test_idxs[i] = self.pad_texts(self.test_idxs[i], self.max_text_len, tokenizer)

    self.train_size = sum([len(train_idx) for train_idx in self.train_idxs])
    self.dev_size = sum([len(dev_idx) for dev_idx in self.dev_idxs])
    self.test_size = sum([len(test_idx) for test_idx in self.test_idxs])

    # deal with rest of training data
    self.rest_train_size = 0
    if scale < 1:
      for i in range(self.label_size):
        # pad
        self.rest_train_idxs[i] = self.pad_texts(self.rest_train_idxs[i], self.max_text_len,
                                                 tokenizer)
      self.rest_train_size = sum([len(rest_train_idx) for rest_train_idx in self.rest_train_idxs])
      # flat the rest of the training, mix labels
      self.flat_rest_train()



  def get_data(self, params, input_texts, tokenizer, train = False, scale = 1.0):
    input_idxs = []
    for key, label in self.idx2label.items():
      input_text = input_texts[label]
      input_idx = [self.encode_text(doc, tokenizer) for doc in input_text]
      # add the data according to label id
      input_idxs.append(input_idx)

    # prepare data
    if train:
      # train
      train_idxs, train_lens, train_max_len, \
      rest_train_idxs, rest_train_lens = self.prepare_data(params, input_idxs, scale)
      train_prop = np.array([len(train_idx) for train_idx in train_idxs])
      train_prop = (train_prop / sum(train_prop)) if sum(train_prop) != 0 else [0] * len(
        train_prop)
      return (train_idxs, train_lens, train_max_len, rest_train_idxs, rest_train_lens, train_prop)
    else:
      dt_idxs, dt_lens, dt_max_len, _, _ = self.prepare_data(params, input_idxs, scale)
      dt_prop = np.array([len(dt_idx) for dt_idx in dt_idxs])
      dt_prop = dt_prop / sum(dt_prop)
      return dt_idxs, dt_lens, dt_max_len, dt_prop


  def encode_text(self, texts, tokenizer):
    text_idx = []
    for line in texts:
      line = self.preprocess(line)
      tokens = tokenizer.tokenize(line)
      line_idx = tokenizer.convert_tokens_to_ids(tokens)
      text_idx += line_idx
    return text_idx


  def pad_texts(self, text_idxs, max_text_len, tokenizer):
    padded_text_idxs = []
    for line_idx in text_idxs:
      padded_line_idx = line_idx[: max_text_len]
      padded_line_idx = padded_line_idx + [tokenizer.vocab['[PAD]']] * (max_text_len - len(padded_line_idx))
      padded_line_idx = [tokenizer.vocab['[CLS]']] + padded_line_idx + [tokenizer.vocab['[SEP]']]
      assert(len(padded_line_idx) == 200)
      padded_text_idxs.append(padded_line_idx)
    return torch.LongTensor(padded_text_idxs)

