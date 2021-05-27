# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Model for CLDC task
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np

import torch
import torch.nn as nn

from xling_embedding_layer import XlingEmbeddingLayer
from encoder import Encoder
from inferer import Inferer
from mlp.cldc_classifier import CLDCClassifier
from xlingva import XlingVA

import pdb


class CLDCModel(nn.Module):
  def __init__(self, params, data_list, classifier_config, model_dict = None):
    super(CLDCModel, self).__init__()
    # embedding layer
    self.embeddings = XlingEmbeddingLayer(params, data_list)
    # encoder
    self.encoder = Encoder(params)
    # inferer
    self.inferer = Inferer(params, params.inf_in_dim)

    # load pretrained model
    if model_dict is not None:
      XlingVA.init_model(self, model_dict)
    
    # CLDC classifier 
    self.cldc_classifier = CLDCClassifier(params, classifier_config)
    
    self.use_cuda = params.cuda
    if self.use_cuda:
      self.cuda()


  def get_gaus(self, lang, batch_in, batch_lens):
    # embedding
    input_word_embs = self.embeddings(lang, batch_in)
    # encoding
    hid = self.encoder(input_word_embs, batch_lens)
    # infering
    mu, logvar = self.inferer(hid)
    return mu, logvar


  def forward(self, lang, batch_in, batch_lens, batch_lb = None, vis = False):
    # embedding
    input_word_embs = self.embeddings(lang, batch_in)
    # encoding
    hid = self.encoder(input_word_embs, batch_lens)
    # infering
    mu, logvar = self.inferer(hid)
    z = self.inferer.reparameterize(mu, logvar)
    # classifier
    loss, pred_p, pred = self.cldc_classifier(z, batch_lb, self.training, vis=vis)
    #loss, pred_p, pred = self.cldc_classifier(hid, batch_lb, self.training, vis = vis)
    return loss, pred_p, pred
