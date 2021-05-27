# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Decoder of vae cross-lingual emb
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class Decoder(nn.Module):
  def __init__(self, params, data_x, data_y):
    super(Decoder, self).__init__()
    # embedding layer
    self.emb_dim = params.emb_dim
    self.embeddings_x = nn.Embedding(data_x.vocab.vocab_size, self.emb_dim, padding_idx = data_x.vocab.PAD_ID)
    self.embeddings_y = nn.Embedding(data_y.vocab.vocab_size, self.emb_dim, padding_idx = data_y.vocab.PAD_ID)
    self.init_emb(params.pretrained, data_x.vocab, data_y.vocab)

    # all vocab idxs, for the lookup of the whole embedding matrix
    self.all_x = torch.LongTensor(range(data_x.vocab.vocab_size))
    self.all_y = torch.LongTensor(range(data_y.vocab.vocab_size))
    
    # bias term
    self.max_sent_len_x = data_x.max_text_len
    self.max_sent_len_y = data_y.max_text_len
    self.bias_x = nn.Parameter(torch.rand(data_x.vocab.vocab_size))
    self.bias_y = nn.Parameter(torch.rand(data_y.vocab.vocab_size))

    self.criterion = nn.CrossEntropyLoss(ignore_index = 0, reduction = 'none')

    self.use_cuda = params.cuda
    if self.use_cuda:
      self.all_x = self.all_x.cuda()
      self.all_y = self.all_y.cuda()


  def init_emb(self, pretrained, vocab_x, vocab_y):
    if pretrained is not None:
      vocab_x.load_pretrained(pretrained, self.embeddings_x)
      vocab_y.load_pretrained(pretrained, self.embeddings_y)
    else:
      # Initialize embedding weight like word2vec.
      # The u_embedding is a uniform distribution in [-0.5/emb_dim, 0.5/emb_dim],
      initrange = 0.5 / self.emb_dim
      self.embeddings_x.weight.data.uniform_(-initrange, initrange)
      self.embeddings_y.weight.data.uniform_(-initrange, initrange)
    self.embeddings_x.weight.data[vocab_x.PAD_ID] = 0
    self.embeddings_y.weight.data[vocab_y.PAD_ID] = 0


  def forward(self, samp_hid, batch_x, batch_y):
    # calculate negative log liklihood
    nll_x = self.cal_nll(samp_hid, batch_x, self.embeddings_x, self.all_x, self.max_sent_len_x, self.bias_x)
    nll_y = self.cal_nll(samp_hid, batch_y, self.embeddings_y, self.all_y, self.max_sent_len_y, self.bias_y)

    # calculate likelihood
    nll = nll_x + nll_y
    nll = torch.mean(nll)

    return nll


  def cal_nll(self, z, x, embeddings, all_idxs, x_len, bias):
    # get all embeddings from embedding matrix
    # |vocab|, emb_dim
    all_emb = embeddings(all_idxs)
    
    # scores before adding biases
    # bs, vocab_size
    all_score = z @ torch.t(all_emb)

    # add dim of sequence length, predict for each position simutaneously
    # bs, x_len, vocab_size
    all_score = all_score.unsqueeze(1).expand(-1, x_len, -1) + bias
    # bs , vocab_size, x_len
    all_score = all_score.permute(0, 2, 1)
    # negtive log likelihood
    # bs, x_len
    nll_loss = self.criterion(all_score, x)
    # sum over the sequence length
    # bs
    nll_loss = torch.sum(nll_loss, dim = 1)

    return nll_loss
