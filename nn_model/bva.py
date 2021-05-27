# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Bilingual variational autoencoder
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoder import Encoder
from inferer import Inferer
from decoder import Decoder
from decoder_lm import LMDecoder

import pdb


class BVA(nn.Module):
  def __init__(self, params, data_x, data_y):
    super(BVA, self).__init__()
    # encoder
    self.encoder_x = Encoder(params, data_x.vocab)
    self.encoder_y = Encoder(params, data_y.vocab)

    # inferer
    self.inferer = Inferer(params)
    # number of samples from z
    self.sample_n = params.sample_n
    self.ls_type = params.ls_type

    # decoder
    self.decoder = Decoder(params, data_x, data_y)

    self.use_cuda = params.cuda
    if self.use_cuda:
      self.cuda()


  def forward(self, batch_x, batch_x_lens, batch_y, batch_y_lens):
    # encoding
    hid_x = self.encoder_x(batch_x, batch_x_lens)
    hid_y = self.encoder_y(batch_y, batch_y_lens)
    # infering
    mu, logvar = self.inferer([hid_x, hid_y])

    if self.ls_type == 'nlldetz':
      # deterministic z
      samp_hid = mu.unsqueeze(1).repeat(1, self.sample_n, 1)
      samp_batch_x = batch_x.unsqueeze(-1).repeat(1, 1, self.sample_n)
      samp_batch_y = batch_y.unsqueeze(-1).repeat(1, 1, self.sample_n)
    else:
      # sample n times from z
      samp_mu = mu.unsqueeze(1).repeat(1, self.sample_n, 1)
      samp_logvar = logvar.unsqueeze(1).repeat(1, self.sample_n, 1)
      samp_hid = self.inferer.reparameterize(samp_mu, samp_logvar)
      samp_batch_x = batch_x.unsqueeze(-1).repeat(1, 1, self.sample_n)
      samp_batch_y = batch_y.unsqueeze(-1).repeat(1, 1, self.sample_n)

    # decoding, neg log likelihood
    nll = self.decoder(samp_hid, samp_batch_x, samp_batch_y)

    # kl divergence, 2 times, considering 2 languages
    kld = 2 * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1))
  
    return nll, kld


  def save_embedding(self, params, data, xory):
    """
    Save the model and the embeddings
    """
    self.eval()
    lang = data.vocab.lang
    word2idx = data.vocab.word2idx
    idx2word = data.vocab.idx2word
    if xory == 'x':
      embeddings = self.decoder.embeddings_x
    else:
      embeddings = self.decoder.embeddings_y
    emb_out = '{}.{}.txt'.format(lang, params.emb_out_path)
    print('Saving {}'.format(emb_out))
    # save embedding
    word_idxs = []
    words = []
    with open(emb_out, 'w') as fout:
      #fout.write('{} {}\n'.format(len(idx2word), params.emb_dim))
      for word_idx, word in idx2word.items():
        word_idxs.append(word_idx)
        words.append(word)
        if len(word_idxs) < params.bs:
          continue
        self.dump_emb_str(embeddings, word_idxs, words, fout)
        word_idxs = []
        words = []
      self.dump_emb_str(embeddings, word_idxs, words, fout)


  def dump_emb_str(self, embeddings, word_idxs, words, fout):
    assert(len(word_idxs) == len(words))
    word_idxs = torch.LongTensor(word_idxs).cuda() if self.use_cuda else torch.LongTensor(word_idxs)
    word_embs = embeddings(word_idxs)
    word_embs = word_embs.data.cpu().numpy().tolist()
    word_embs = list(zip(words, word_embs))
    word_embs = ['{} {}'.format(w[0], ' '.join(list(map(lambda x: str(x), w[1])))) for w in word_embs]
    fout.write('{}\n'.format('\n'.join(word_embs)))


  def save_model(self, params, data_x, data_y, optimizer):
    self.eval()
    model_out = '{}.pth'.format(params.emb_out_path)
    print('Saving {}'.format(model_out)) 
    # save model
    model_dict = {
                 'model': self.state_dict(), 
                 'optimizer': optimizer.state_dict(),
                 'vocab_x': data_x.vocab,
                 'vocab_y': data_y.vocab
                 }
    torch.save(model_dict, model_out) 
