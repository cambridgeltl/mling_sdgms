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

from decoder import Decoder

import pdb


class XlingDecoder(Decoder):
  def __init__(self, params, datas, embeddings):
    super(Decoder, self).__init__()
    self.lang_dict = params.lang_dict
    self.use_cuda = params.cuda

    if params.dec_type == 'bow':
      self.init_bow(params, datas, embeddings)
    elif params.dec_type == 'gru':
      self.in_dim = params.dec_rnn_in_dim
      self.hid_dim = params.dec_rnn_hid_dim
      self.init_gru(params, datas, embeddings)

    self.criterion = nn.CrossEntropyLoss(ignore_index = 0, reduction = 'none')
    self.forward = getattr(self, 'forward_{}'.format(params.dec_type))


  def init_bow(self, params, datas, embeddings):
    # tie with encoder embeddings
    self.embeddings = embeddings
    # all vocab idxs, for the lookup of the whole embedding matrix
    self.all_idxs = []
    self.max_sent_lens = []
    # bias term
    self.biases = nn.ParameterList([])
    for i, data in enumerate(datas):
      vocab = data.vocab
      #assert(self.lang_dict[vocab.lang] == i)
      self.all_idxs.append(torch.LongTensor(range(vocab.vocab_size)))
      self.max_sent_lens.append(data.max_text_len)
      self.biases.append(nn.Parameter(torch.rand(vocab.vocab_size)))

    if self.use_cuda:
      for i in range(len(self.all_idxs)):
        self.all_idxs[i] = self.all_idxs[i].cuda()


  def init_gru(self, params, datas, embeddings):
    # tie with encoder embeddings
    self.embeddings = embeddings

    self.z2hid = nn.ModuleList([])
    self.zx2decin = nn.ModuleList([])
    self.rnns = nn.ModuleList([])
    self.hid2vocab = nn.ModuleList([])

    if params.single_dec is True:
      self.z2hid.append(nn.Linear(params.z_dim, self.hid_dim))
      self.zx2decin.append(nn.Linear(params.z_dim + params.emb_dim, self.in_dim))
      self.rnns.append(nn.GRU(self.in_dim,
                         self.hid_dim,
                         batch_first = True))
      # the only language specific thing
      for i, data in enumerate(datas):
        vocab = data.vocab
        self.hid2vocab.append(nn.Linear(self.hid_dim, vocab.vocab_size))
    else:
      for i, data in enumerate(datas):
        vocab = data.vocab
        self.z2hid.append(nn.Linear(params.z_dim, self.hid_dim))
        self.zx2decin.append(nn.Linear(params.z_dim + params.emb_dim, self.in_dim))
        self.rnns.append(nn.GRU(self.in_dim,
                                self.hid_dim,
                                batch_first = True))
        self.hid2vocab.append(nn.Linear(self.hid_dim, vocab.vocab_size))


  def forward_bow(self, lang, z, x, reduction = 'mean'):
    lang_idx = self.lang_dict[lang] 

    nll_loss = super(XlingDecoder, self).cal_nll(z, x, 
                                                 self.embeddings.get_lang_emb(lang), 
                                                 self.all_idxs[lang_idx], 
                                                 self.max_sent_lens[lang_idx], 
                                                 self.biases[lang_idx]) 
    # pytorch like reduction
    if reduction is None:
      return nll_loss
    elif reduction == 'sum':
      return torch.sum(nll_loss)
    elif reduction == 'mean':
      return torch.mean(nll_loss)


  def forward_gru(self, lang, z, x, reduction = 'mean'):
    lang_idx = self.lang_dict[lang]

    # random z
    #z = torch.rand_like(z)
    # random z

    # single_gru
    if len(self.rnns) == 1:
      # the first hidden state
      dec_init_hid = self.z2hid[0](z).unsqueeze(0)

      # add pad as the start symbol, then remove the last token
      dec_init_in = torch.zeros(dec_init_hid.shape[1], dtype=torch.long).unsqueeze(1)
      if self.use_cuda:
        dec_init_in = dec_init_in.cuda()
      dec_ins = torch.cat((dec_init_in, x), dim=1)[:, :x.shape[-1]]
      dec_in_embs = self.embeddings.embeddings[lang_idx](dec_ins)

      # concatenate with z
      dec_in_embs = torch.cat((dec_in_embs, z.unsqueeze(1).repeat(1, dec_in_embs.shape[1], 1)), dim=-1)
      # linear transformation
      dec_in_embs = self.zx2decin[0](dec_in_embs)

      # full teacher forcing
      out, hid = self.rnns[0](dec_in_embs, dec_init_hid)

    elif len(self.rnns) > 1:
      # the first hidden state
      dec_init_hid = self.z2hid[lang_idx](z).unsqueeze(0)

      # add pad as the start symbol, then remove the last token
      dec_init_in = torch.zeros(dec_init_hid.shape[1], dtype = torch.long).unsqueeze(1)
      if self.use_cuda:
        dec_init_in = dec_init_in.cuda()
      dec_ins = torch.cat((dec_init_in, x), dim = 1)[:, :x.shape[-1]]
      dec_in_embs = self.embeddings.embeddings[lang_idx](dec_ins)

      # concatenate with z
      dec_in_embs = torch.cat((dec_in_embs, z.unsqueeze(1).repeat(1, dec_in_embs.shape[1], 1)), dim = -1)
      # linear transformation
      dec_in_embs = self.zx2decin[lang_idx](dec_in_embs)

      # full teacher forcing
      out, hid = self.rnns[lang_idx](dec_in_embs, dec_init_hid)

    scores = self.hid2vocab[lang_idx](out).permute(0, 2, 1)

    nll_loss = self.criterion(scores, x)

    nll_loss = torch.sum(nll_loss, dim = 1)

    # pytorch like reduction
    if reduction is None:
      return nll_loss
    elif reduction == 'sum':
      return torch.sum(nll_loss)
    elif reduction == 'mean':
      return torch.mean(nll_loss)
