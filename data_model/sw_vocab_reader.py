# -*- coding: UTF-8 -*-
# !/usr/bin/python3
"""
Subword Vocab Reader
"""

# ************************************************************
# Imported Libraries
# ************************************************************
import os
from collections import defaultdict
from tqdm import tqdm
import sentencepiece as spm

from train_bert.tokenization import convert_to_unicode, whitespace_tokenize, convert_by_vocab


class SPMVocabReader(object):
  def __init__(self, params, lang, PAD = '<pad>', UNK = '<unk>'):
    # languages
    self.lang = lang
    lang_idx = params.lang_dict[lang]

    # maximum vocab size, considering PAD (UNK is automatically in the vocab)
    self.max_vocab_size = params.vocab_sizes[lang_idx] + 1

    # pad and unk
    self.PAD = PAD
    self.PAD_ID = 0
    self.UNK = UNK
    self.tok2idx = {self.PAD: self.PAD_ID}
    self.idx2tok = {self.PAD_ID: self.PAD}

    # (word freq) dict file
    self.vocab_file = params.vocab_path[lang_idx]
    assert(os.path.exists(self.vocab_file))

    # read data, build vocab table
    self.build_vocab()
    self.vocab_size = len(self.tok2idx)

    # tokenizer
    self.seg_model_path = params.seg_model_path
    self.tokenizer = spm.SentencePieceProcessor()
    self.tokenizer.Load(self.seg_model_path)


  @property
  def UNK_ID(self):
    return self.tok2idx[self.UNK]


  def build_vocab(self):
    """
    build vocab table
    """
    with open(self.vocab_file, 'r') as fin:
      for line in fin:
        linevec = convert_to_unicode(line).strip().split('\t')
        assert (len(linevec) == 2 and linevec[0] not in self.tok2idx)
        w = linevec[0]
        idx = len(self.tok2idx)
        self.tok2idx[w] = idx
        self.idx2tok[idx] = w
    assert(len(self.tok2idx) == self.max_vocab_size)


  def tokenize(self, text):
    return [t if t in self.tok2idx else self.UNK for t in self.tokenizer.EncodeAsPieces(text)]


  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.tok2idx, tokens)


  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.idx2tok, ids)


  # spm.SentencePieceProcessor() is a SwigPyObject object which cannot be
  # pickled. We need to define __getstate__ here.
  def __getstate__(self):
    state = self.__dict__.copy()
    state["tokenizer"] = None
    return state

  # spm.SentencePieceProcessor() is a SwigPyObject object which cannot be
  # pickled. We need to define __setstate__ here.
  def __setstate__(self, d):
    self.__dict__ = d
    self.tokenizer = spm.SentencePieceProcessor()
    self.tokenizer.Load(self.seg_model_path)

