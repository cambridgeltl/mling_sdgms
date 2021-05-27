# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Data Reader
"""

#************************************************************
# Imported Libraries
#************************************************************
import os
import sys
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch

import pdb


class DataReader(object):
  def __init__(self, params, vocab):
    self.vocab = vocab
    lang_idx = params.lang_dict[vocab.lang]
    
    self.train_path = params.train_path[lang_idx]
    self.dev_path = params.dev_path[lang_idx]
    self.train_idxs, self.train_lens, self.max_train_len = self.get_data(params, self.train_path, scale = params.train_scale)
    self.train_size = len(self.train_idxs)
    self.dev_idxs, self.dev_lens, self.max_dev_len = self.get_data(params, self.dev_path)
    self.dev_size = len(self.dev_idxs)
    # max sentence length
    self.max_text_len = max(self.max_train_len, self.max_dev_len)


  def get_data(self, params, input_path, scale = 1):
    id_file = input_path + '.id'
    if os.path.exists(id_file):
      # found binary data
      print('Found ID data in {}, loading {}'.format(self.vocab.lang, id_file))
      data_dict = torch.load(id_file)
    else:
      print('Did not find ID data in {}, reading {}'.format(self.vocab.lang, input_path))
      data_dict = self.read_data(params, input_path)

    n_inst = int(len(data_dict['text_idxs']) * scale)
    data_dict['text_idxs'] = data_dict['text_idxs'][:n_inst]
    data_dict['text_lens'] = data_dict['text_lens'][:n_inst]

    # cut longer sentences
    #max_text_len = int(np.max(data_dict['text_lens']))
    # use threshold of percentile
    #max_text_len = int(np.percentile(data_dict['text_lens'], params.corpus_percentile))
    max_text_len = 200
    #print(max_text_len)
    for i, t_idx in enumerate(data_dict['text_idxs']):
      data_dict['text_idxs'][i] = t_idx[:max_text_len]
      data_dict['text_lens'][i] = min(data_dict['text_lens'][i], max_text_len)

    return data_dict['text_idxs'], data_dict['text_lens'], max_text_len


  def read_data(self, params, input_path):
    with open(input_path, 'r') as fin:
      texts = fin.readlines()

    # get text idx, text lens
    text_idxs, text_lens = self.encode_text(texts) 
    data_dict = {
        'text_idxs': text_idxs,
        'text_lens': text_lens,
        }

    # save idxs 
    print('Saving ID data...')
    torch.save(data_dict, input_path + '.id')

    return data_dict


  def encode_text(self, texts):
    text_idxs = []
    text_lens = []
    for line in tqdm(texts):
      linevec = line.strip().split(' ')
      line_idx = [self.vocab.word2idx[w] if w in self.vocab.word2idx else self.vocab.UNK_ID for w in linevec]
      text_idxs.append(line_idx)
      text_lens.append(len(line_idx))

    return text_idxs, np.array(text_lens)


  def idx2text(self, idxs, idx_lens = None):
    # if idx_lens == None, idxs is not padded, otherwise idxs is padded
    idxs = [l[:] for i, l in enumerate(idxs)] if idx_lens is None else [l[:idx_lens[i]] for i, l in enumerate(idxs)]
    text = ['_'.join([self.vocab.idx2word[idx] if idx in self.vocab.idx2word else self.vocab.UNK for idx in l]) for l in idxs]
    return text

