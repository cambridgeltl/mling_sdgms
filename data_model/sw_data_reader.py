# -*- coding: UTF-8 -*-
# !/usr/bin/python3
"""
SPM Data Reader
"""

# ************************************************************
# Imported Libraries
# ************************************************************
import os
import sys
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch

from train_bert.tokenization import convert_to_unicode
from data_reader import DataReader


class SPMDataReader(DataReader):
  def __init__(self, params, vocab):
    DataReader.__init__(self, params, vocab)


  def encode_text(self, texts):
    text_idxs = []
    text_lens = []
    for line in tqdm(texts):
      line = line.strip()
      line_tok = self.vocab.tokenize(line)
      line_idx = self.vocab.convert_tokens_to_ids(line_tok)
      assert(len(line_tok) == len(line_idx))
      text_idxs.append(line_idx)
      text_lens.append(len(line_idx))

    return text_idxs, np.array(text_lens)
