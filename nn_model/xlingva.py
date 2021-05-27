# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
a much Better Cross-lingual variational autoencoder :)
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bva import BVA
from xling_embedding_layer import XlingEmbeddingLayer
from encoder import Encoder
from mlp.xlingdiscriminator import XlingDiscriminator
from inferer import Inferer 
from xlingdecoder import XlingDecoder
from data_model.cldc_data_reader import CLDCDataReader
from utils.legacy import xlingva_param_map
import copy

import pdb


class XlingVA(BVA):
  def __init__(self, params, datas, model_dict):
    super(BVA, self).__init__()

    # embedding layer
    self.embeddings = XlingEmbeddingLayer(params, datas)
    # encoder
    self.encoder = Encoder(params)
    
    # discriminator for adversarial training
    self.adv_training = params.adv_training
    if self.adv_training:
      self.discriminator = XlingDiscriminator(params)

    # inferer
    self.inferer = Inferer(params, params.inf_in_dim)

    # vae type
    self.vae_type = params.vae_type

    # decoder
    # untie the embeddings
    if params.tie_emb:
      self.decoder = XlingDecoder(params, datas, self.embeddings)
    else:
      embeddings = copy.deepcopy(self.embeddings)
      """
      # random initialization
      for emb in embeddings.embeddings:
        initrange = 0.5 / 300
        emb.embeddings.weight.data.uniform_(-initrange, initrange)
        emb.embeddings.weight.data[0] = 0
      """
      self.decoder = XlingDecoder(params, datas, embeddings)

    # load pretrained model
    if model_dict is not None:
      self.init_model(self, model_dict)

    # load pretrained embeddings again in case we need
    if model_dict is not None and params.pretrained_emb_path[0] is not None:
      for i, lang in enumerate(self.embeddings.lang_dict):
        if params.pretrained_emb_path[i] is not None:
          vocab = datas[i].vocab
          assert (lang == vocab.lang)
          self.embeddings.embeddings[i].init_emb(
            self.embeddings.embeddings[i].embeddings,
            params.pretrained_emb_path[i],
            vocab)

    self.use_cuda = params.cuda
    if self.use_cuda:
      self.cuda()


  @staticmethod 
  def init_model(obj, model_dict):
    # get the current module parameter dicts
    cur_model_dict = obj.state_dict()
    # 1. filter out unnecessary keys
    filtered_model_dict = {}
    try:
      # old code model
      for k, v in model_dict.items():
        if xlingva_param_map[k] not in cur_model_dict:
          continue
        filtered_model_dict[xlingva_param_map[k]] = v
    except:
      # new code model
      filtered_model_dict = {k: model_dict[k] for k in model_dict if k in cur_model_dict}

    # only for MVAE in CLDC de case
    emb_dict = [k for k in filtered_model_dict if 'embedding' in k]
    if filtered_model_dict[emb_dict[0]].shape[0] == 50002:
      filtered_model_dict['embeddings.embeddings.1.embeddings.weight'] = filtered_model_dict['embeddings.embeddings.0.embeddings.weight']
      del filtered_model_dict['embeddings.embeddings.0.embeddings.weight']
    # only for MVAE in CLDC de case

    #assert(set(filtered_model_dict.keys()) == set(cur_model_dict.keys()))
    # 2. overwrite entries in the existing state dict
    cur_model_dict.update(filtered_model_dict)
    # 3. load the new state dict
    obj.load_state_dict(cur_model_dict)


  def get_hidx(self, lang, batch_in, batch_lens):
    input_word_embs = self.embeddings(lang, batch_in)
    # encoding
    hid = self.encoder(input_word_embs, batch_lens)

    return hid


  def get_gaus(self, lang, batch_in, batch_lens):
    # get hidden rep of x
    hid = self.get_hidx(lang, batch_in, batch_lens)

    # infering
    mu, logvar = self.inferer(hid)

    loss_dis, loss_enc = torch.tensor(.0), torch.tensor(.0)
    if self.adv_training:
      # discriminator
      loss_dis, loss_enc = self.discriminator(lang, hid)

    return mu, logvar, hid, loss_dis, loss_enc


  def forward(self, lang, batch_in, batch_lens): 
    # get hidden rep of x
    hid = self.get_hidx(lang, batch_in, batch_lens)

    loss_dis, loss_enc = None, None
    if self.adv_training:
      # discriminator
      loss_dis, loss_enc = self.discriminator(lang, hid)

    # infering
    mu, logvar = self.inferer(hid)

    if self.vae_type == 'nlldetz':
      # deterministic z
      samp_hid = mu.unsqueeze(1).repeat(1, 1, 1)
      samp_batch_in = batch_in.unsqueeze(-1).repeat(1, 1, 1)
    else:
      # vae
      hid_rec = self.inferer.reparameterize(mu, logvar)

    # decoding
    # neg log likelihood
    nll = self.decoder(lang, hid_rec, batch_in)

    # kl divergence
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1))
  
    return nll, kld, loss_dis, loss_enc
  

  def save_embedding(self, params, data):
    self.eval()
    lang = data.vocab.lang
    if params.seg_type == 'word':
      word2idx = data.vocab.word2idx
      idx2word = data.vocab.idx2word
    elif params.seg_type == 'spm':
      word2idx = data.vocab.tok2idx
      idx2word = data.vocab.idx2tok

    embeddings = self.embeddings.get_lang_emb(lang)
    emb_out = '{}.{}.txt'.format(lang, params.log_path)
    print('Saving {}'.format(emb_out))
    # save embedding
    word_idxs = []
    words = []
    with open(emb_out, 'w') as fout:
      #fout.write('{} {}\n'.format(len(idx2word), params.emb_dim))
      for word_idx, word in idx2word.items():
        word_idxs.append(word_idx)
        words.append(word)
        if len(word_idxs) < params.test_bs:
          continue
        super(XlingVA, self).dump_emb_str(embeddings, word_idxs, words, fout)
        word_idxs = []
        words = []
      super(XlingVA, self).dump_emb_str(embeddings, word_idxs, words, fout)


  def save_model(self, params, datas):
    self.eval()
    model_out = '{}.pth'.format(params.log_path)
    model_dict = {'model': {k: v.cpu() for k, v in self.state_dict().items()}}
    # save model
    for lang in params.langs:
      lang_idx = params.lang_dict[lang]
      model_dict[lang] = datas[lang_idx].vocab
    torch.save(model_dict, model_out)
