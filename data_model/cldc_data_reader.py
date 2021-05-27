# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Cldc Task Data Reader
"""

#************************************************************
# Imported Libraries
#************************************************************
import os
import sys
from tqdm import tqdm
from collections import Counter
import random

import numpy as np
import torch

from vocab_reader import VocabReader
from classification_base_data_reader import CLSBaseDataReader

import pdb


class CLDCDataReader(CLSBaseDataReader):
  def __init__(self, params, lang_idx, vocab):
    super(CLDCDataReader, self).__init__(vocab) 

    # cldc data path 
    self.cldc_path = params.cldc_path[lang_idx]
    assert(os.path.exists(self.cldc_path))
    texts = torch.load(self.cldc_path)
    #train_texts, dev_texts, test_texts = texts['train'], texts['dev'], texts['test']
    train_texts, dev_texts, test_texts = texts['train.1000'], texts['dev'], texts['test']
    for l in params.cldc_label2idx:
      train_texts[l] = [d.strip().split('\n') for d in train_texts[l]]
      dev_texts[l] = [d.strip().split('\n') for d in dev_texts[l]]
      test_texts[l] = [d.strip().split('\n') for d in test_texts[l]]

    # cldc labels
    self.label2idx = params.cldc_label2idx
    self.idx2label = params.cldc_idx2label
    self.label_size = params.cldc_label_size
    
    # read train
    # different scale for crosslingual training data
    if params.seg_type == 'word':
      scale = params.cldc_train_scale if vocab.lang == params.cldc_langs[0] else params.cldc_xl_train_scale
    elif params.seg_type == 'spm':
      scale = params.cldc_train_scale
    self.train_idxs, self.train_lens, self.max_train_len, \
    self.rest_train_idxs, self.rest_train_lens, self.train_prop = self.get_data(params, train_texts, train = True, scale = scale)
    # read dev data
    self.dev_idxs, self.dev_lens, self.max_dev_len, self.dev_prop = self.get_data(params, dev_texts)
    # read test data
    self.test_idxs, self.test_lens, self.max_test_len, self.test_prop = self.get_data(params, test_texts) 
    
    # get the max of sentence length, then pad accroding to max_text_len
    self.max_text_len = max(self.max_train_len, self.max_dev_len, self.max_test_len)
    for i in range(self.label_size):
      self.train_idxs[i] = self.pad_texts(self.train_idxs[i], self.max_text_len, self.vocab.PAD_ID)
      self.dev_idxs[i] = self.pad_texts(self.dev_idxs[i], self.max_text_len, self.vocab.PAD_ID)
      self.test_idxs[i] = self.pad_texts(self.test_idxs[i], self.max_text_len, self.vocab.PAD_ID)

    self.train_size = sum([len(train_idx) for train_idx in self.train_idxs])
    self.dev_size = sum([len(dev_idx) for dev_idx in self.dev_idxs])
    self.test_size = sum([len(test_idx) for test_idx in self.test_idxs])
   
    # deal with rest of training data
    self.rest_train_size = 0
    if scale < 1:
      for i in range(self.label_size):
        # pad
        self.rest_train_idxs[i] = self.pad_texts(self.rest_train_idxs[i], self.max_text_len, self.vocab.PAD_ID)
      self.rest_train_size = sum([len(rest_train_idx) for rest_train_idx in self.rest_train_idxs])
      # flat the rest of the training, mix labels
      self.flat_rest_train()
