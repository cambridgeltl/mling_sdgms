# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Classification task base data reader
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import torch


import pdb


class CLSBaseDataReader(object):
  def __init__(self, vocab):
    super().__init__()
    # vocab
    self.vocab = vocab


  def get_data(self, params, input_texts, train = False, scale = 1.0):
    input_idxs = []
    for key, label in self.idx2label.items():
      input_text = input_texts[label]
      if params.seg_type == 'word':
        input_idx = [self.encode_text(doc) for doc in input_text]
      elif params.seg_type == 'spm':
        input_idx = [self.encode_text_spm(doc) for doc in input_text]
      # add the data according to label id
      input_idxs.append(input_idx) 

    # prepare data
    if train:
      '''
      # 23lb
      if 'cldc' in params.task:
        if self.vocab.lang == 'en':
          input_idxs = [input_idx[:2870] for input_idx in input_idxs]
        elif self.vocab.lang == 'de':
          input_idxs = [input_idx[:2870] for input_idx in input_idxs]
          #input_idxs = [input_idx[:3862] for input_idx in input_idxs]
      # 23lb
      '''
      # train
      train_idxs, train_lens, train_max_len, \
      rest_train_idxs, rest_train_lens = self.prepare_data(params, input_idxs, scale)
      train_prop = np.array([len(train_idx) for train_idx in train_idxs])
      train_prop = (train_prop / sum(train_prop)) if sum(train_prop) != 0 else [0] * len(train_prop)
      return (train_idxs, train_lens, train_max_len, rest_train_idxs, rest_train_lens, train_prop)
    else:
      '''
      # 23lb
      if 'cldc' in params.task:
        if self.vocab.lang == 'en':
          input_idxs = [input_idx[:1650] if len(input_idx) >= 1650 else input_idx[:318] for input_idx in input_idxs]
        elif self.vocab.lang == 'de':
          input_idxs = [input_idx[:1650] if len(input_idx) >= 1650 else input_idx[:318] for input_idx in input_idxs]
          #input_idxs = [input_idx[:2031] if len(input_idx) >= 2031 else input_idx[:429] for input_idx in input_idxs]
      # 23lb
      '''
      dt_idxs, dt_lens, dt_max_len, _, _ = self.prepare_data(params, input_idxs, scale)
      dt_prop = np.array([len(dt_idx) for dt_idx in dt_idxs])
      dt_prop = dt_prop / sum(dt_prop)
      return dt_idxs, dt_lens, dt_max_len, dt_prop


  def encode_text(self, texts):
    text_idx = []
    for line in texts:
      line = self.preprocess(line)
      linevec = line.split(' ')
      line_idx = [self.vocab.word2idx[w] if w in self.vocab.word2idx else self.vocab.UNK_ID for w in linevec]
      text_idx += line_idx

    return text_idx


  def encode_text_spm(self, texts):
    text_idx = []
    for line in texts:
      line = self.preprocess(line)
      line_tok = self.vocab.tokenize(line)
      line_idx = self.vocab.convert_tokens_to_ids(line_tok)
      assert(len(line_tok) == len(line_idx))
      text_idx += line_idx

    return text_idx


  def preprocess(self, line):
    # same preprocessing as in the original paper
    line = line.strip()
    line = line.lower()
    return line


  def prepare_data(self, params, input_idxs, scale):
    prop_input_idxs = []
    rest_input_idxs = []
    # search and select
    if params.ss_file:
      idx_map = torch.load(params.ss_file)
      idx_map = {0: idx_map[0]}
      #idx_map = {0: idx_map[0], 1: idx_map[1]}
      #idx_map = {0: idx_map[0], 1: idx_map[1], 3: idx_map[3]}
      #idx_map = {0: idx_map[0], 1: idx_map[1], 2: idx_map[2], 3: idx_map[3]}
    # search and select
    for i, input_idx in enumerate(input_idxs):
      # search and select
      if params.ss_file and self.vocab.lang == params.cldc_langs[0] and i in idx_map and scale < 1.0:
        assert(idx_map[i].shape[-1] == len(input_idx))
        prop_input_idx = [input_idx[idx] for idx in idx_map[i][0][:int(scale * len(input_idx))]]
      else:
        # scaled input for each label
        #prop_input_idx = input_idx[: int(scale * len(input_idx))]
        # deal with training de for cldc, 32
        if scale == 0.0333 and i == 1:
          prop_input_idx = input_idx[: int(scale * len(input_idx)) + 1]
        else:
          prop_input_idx = input_idx[: int(scale * len(input_idx))]

      # search and select
      prop_input_idxs.append(prop_input_idx)
      if scale < 1.0:
        # search and select
        if params.ss_file and self.vocab.lang == params.cldc_langs[0] and i in idx_map and scale < 1.0:
          rest_input_idx = [input_idx[idx] for idx in range(len(input_idx)) if idx not in idx_map[i][0][:int(scale * len(input_idx))]]
        else:
          # rest of the input for semi-supervised learning
          rest_input_idx = input_idx[int(scale * len(input_idx)):]
        # search and select
        rest_input_idxs.append(rest_input_idx)
    prop_input_lens = [np.array([len(in_idx) for in_idx in input_idx]) for input_idx in prop_input_idxs]
    rest_input_lens = [np.array([len(in_idx) for in_idx in input_idx]) for input_idx in rest_input_idxs]

    # only consider the proportion used for training
    #max_input_len = max(input_lens)
    max_input_len = self.get_max_input_len(params, prop_input_lens)

    for i, input_idx in enumerate(prop_input_idxs):
      for j, t_idx in enumerate(input_idx):
        prop_input_idxs[i][j] = t_idx[:max_input_len]
        prop_input_lens[i][j] = min(prop_input_lens[i][j], max_input_len)
    for i, input_idx in enumerate(rest_input_idxs):
      for j, t_idx in enumerate(input_idx):
        rest_input_idxs[i][j] = t_idx[:max_input_len]
        rest_input_lens[i][j] = min(rest_input_lens[i][j], max_input_len)
    return prop_input_idxs, prop_input_lens, max_input_len, rest_input_idxs, rest_input_lens


  def get_max_input_len(self, params, data_lens):
    max_input_len = 200
    return max_input_len


  def pad_texts(self, text_idxs, max_text_len, PAD_ID):
    padded_text_idxs = []
    for line_idx in text_idxs:
      padded_line_idx = line_idx + [PAD_ID] * (max_text_len - len(line_idx))
      padded_text_idxs.append(padded_line_idx)
    return torch.LongTensor(padded_text_idxs)


  def flat_rest_train(self):
    self.rest_train_idxs = np.concatenate(self.rest_train_idxs)
    self.rest_train_lens = np.concatenate(self.rest_train_lens)
    assert(self.rest_train_idxs.shape[0] == self.rest_train_size)


  def idx2text(self, idxs, batch_lb, idx_lens = None):
    assert(len(idxs) == len(batch_lb))
    # if idx_lens == None, idxs is not padded, otherwise idxs is padded
    idxs = [l[:] for i, l in enumerate(idxs)] if idx_lens is None else [l[:idx_lens[i]] for i, l in enumerate(idxs)]
    text = ['_'.join([self.vocab.idx2word[idx] if idx in self.vocab.idx2word else self.vocab.UNK for idx in l]) + '_{}'.format(batch_lb[i]) for i, l in enumerate(idxs)]
    return text
