# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
MLSA Task Data Reader
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

from classification_base_data_reader import CLSBaseDataReader

import pdb


class MLSADataReader(CLSBaseDataReader):
  def __init__(self, params, vocab):
    super(MLSADataReader, self).__init__(vocab)

    # mlsa data path 
    lang_idx = params.lang_dict[vocab.lang]
    self.mlsa_path = params.mlsa_path[lang_idx]
    assert(os.path.exists(self.mlsa_path))
    texts = torch.load(self.mlsa_path)
    train_texts, dev_texts, test_texts, ub_texts = texts['train'], texts['dev'], texts['test'], texts['ub']

    # mlsa labels
    self.label2idx = params.mlsa_label2idx
    self.idx2label = params.mlsa_idx2label
    self.label_size = params.mlsa_label_size
   
    # read train
    self.train_idxs, self.train_lens, self.max_train_len, \
    self.rest_train_idxs, self.rest_train_lens, self.train_prop = self.get_data(params, train_texts, train = True, scale = params.mlsa_train_scale)
    # read dev data
    self.dev_idxs, self.dev_lens, self.max_dev_len, self.dev_prop = self.get_data(params, dev_texts)
    # read test data
    self.test_idxs, self.test_lens, self.max_test_len, self.test_prop = self.get_data(params, test_texts) 
    # read unlabeled data
    #self.ub_idxs, self.ub_lens, self.max_ub_len, _ = self.get_data(params, ub_texts)

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
    if params.mlsa_train_scale < 1:
      for i in range(self.label_size):
        # pad
        self.rest_train_idxs[i] = self.pad_texts(self.rest_train_idxs[i], self.max_text_len, self.vocab.PAD_ID)
      self.rest_train_size = sum([len(rest_train_idx) for rest_train_idx in self.rest_train_idxs])
      # flat the rest of the training, mix labels
      self.flat_rest_train()

