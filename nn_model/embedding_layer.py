# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Embedding Layer
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import KeyedVectors

import pdb


class EmbeddingLayer(nn.Module):
  def __init__(self, params, vocab, pretrained_emb_path = None):
    super(EmbeddingLayer, self).__init__()
    # embedding layer
    self.lang = vocab.lang
    self.vocab = vocab
    self.emb_dim = params.emb_dim
    self.embeddings = nn.Embedding(vocab.vocab_size, self.emb_dim, padding_idx = vocab.PAD_ID)
    self.init_emb(self.embeddings, pretrained_emb_path, vocab)
    # ijcai dropout, p = 0.2
    self.emb_do = nn.Dropout(p = params.emb_do) 

    self.use_cuda = params.cuda


  def init_emb(self, embeddings, pretrained_emb_path, vocab):
    if pretrained_emb_path is not None:
      self.load_pretrained(pretrained_emb_path, embeddings, vocab)
    else:
      """
      Initialize embedding weight like word2vec.
      The u_embedding is a uniform distribution in [-0.5/emb_dim, 0.5/emb_dim],
      """
      initrange = 0.5 / self.emb_dim
      embeddings.weight.data.uniform_(-initrange, initrange)
    embeddings.weight.data[vocab.PAD_ID] = 0
 

  def load_pretrained(self, pretrained_emb_path, embeddings, vocab):
    print('loading {} embeddings for {}'.format(pretrained_emb_path, self.lang))
    try:
      pre_emb = KeyedVectors.load_word2vec_format(pretrained_emb_path, binary = False)
    except:
      print('Did not found {} embeddings for {}'.format(pretrained_emb_path, self.lang))
      return
    # ignore only pad
    for i in range(1, len(vocab.idx2word)):
      try:
        embeddings.weight.data[i] = torch.from_numpy(pre_emb[vocab.idx2word[i]])
      except:
        continue


  def forward(self, batch_input):
    input_word_embs = self.embeddings(batch_input)
    input_word_embs = self.emb_do(input_word_embs)

    return input_word_embs
