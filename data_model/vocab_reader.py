# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Vocab Reader
"""

#************************************************************
# Imported Libraries
#************************************************************
import os
import sys
from collections import defaultdict
from tqdm import tqdm

import pdb


class VocabReader(object):
  def __init__(self, params, lang, PAD = '<PAD>', UNK = '<UNK>'):
    # language
    self.lang = lang 
    lang_idx = params.lang_dict[lang]

    # maximum vocab size, considering PAD and UNK
    self.max_vocab_size = params.vocab_sizes[lang_idx] + 2 

    # pad and unk
    self.PAD = PAD
    self.PAD_ID = 0
    self.UNK = UNK
    self.UNK_ID = 1
    self.word2idx = {self.PAD: self.PAD_ID, self.UNK: self.UNK_ID}
    self.idx2word = {self.PAD_ID: self.PAD, self.UNK_ID: self.UNK}

    # (word freq) dict file
    self.vocab_file = params.vocab_path[lang_idx]
    # generate word -> freq vocab_file
    if not os.path.exists(self.vocab_file):
      print('Did not found vocabulary file in {}, generating vocabulary file...'.format(self.lang))
      self.gen_vocab(params.train_path[lang_idx])

    # read data, build vocab table
    self.build_vocab()
    self.vocab_size = len(self.word2idx)

  
  def gen_vocab(self, train_path):
    """
    generate vocab: freq dictionary to vocab_file
    """
    word2ct = defaultdict(int)
    line_n = len(open(train_path, 'r').readlines())
    with open(train_path, 'r') as fin:
      for i in tqdm(range(line_n)):
        line = fin.readline()
        linevec = line.strip().split(' ')
        for w in linevec:
          word2ct[w] += 1
    with open(self.vocab_file, 'w') as fout:
      # sort the pair in descending order
      for w, c in sorted(word2ct.items(), key = lambda x: x[1], reverse = True):
        fout.write('{}\t{}\n'.format(w, c))


  def build_vocab(self):
    """
    build vocab table
    """
    with open(self.vocab_file, 'r') as fin:
      for line in fin:
        linevec = line.strip().split('\t')
        assert(len(linevec) == 2 and linevec[0] not in self.word2idx)
        w = linevec[0]
        idx = len(self.word2idx)
        self.word2idx[w] = idx
        self.idx2word[idx] = w
        if len(self.word2idx) == self.max_vocab_size:
          break

def build_inter_vocab_mapping(vocabs):
  assert(len(vocabs) == 2)
  wordtoidx1 = vocabs[0].word2idx
  wordtoidx2 = vocabs[1].word2idx
  # mapping of the index
  vocabs[0].inter_vocab_map = {}
  vocabs[1].inter_vocab_map = {}
  for w, i in wordtoidx1.items():
    if w in wordtoidx2:
      i2 = wordtoidx2[w]
      vocabs[0].inter_vocab_map[i] = i2
      vocabs[1].inter_vocab_map[i2] = i
  assert(len(vocabs[0].inter_vocab_map) == len(vocabs[1].inter_vocab_map))
