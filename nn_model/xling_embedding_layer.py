# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Xlingual Embedding Layer
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_layer import EmbeddingLayer

import pdb


class XlingEmbeddingLayer(nn.Module):
  def __init__(self, params, datas):
    super(XlingEmbeddingLayer, self).__init__()
    self.lang_dict = params.lang_dict
    # embedding list
    self.embeddings = nn.ModuleList([])
    for i, lang in enumerate(self.lang_dict):
      vocab = datas[i].vocab
      assert(lang == vocab.lang)
      self.embeddings.append(EmbeddingLayer(params, vocab, pretrained_emb_path = params.pretrained_emb_path[i]))
 
    self.use_cuda = params.cuda 


  def get_lang_emb(self, lang):
    return self.embeddings[self.lang_dict[lang]]


  def forward(self, lang, batch_input):
    try:
      return self.embeddings[self.lang_dict[lang]](batch_input)
    except:
      return self.embeddings[0](batch_input)
