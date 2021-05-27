# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
CLDC task classifier
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np

import torch
import torch.nn as nn

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from .base_mlp import BaseMLP

import pdb


class CLDCClassifier(nn.Module):
  def __init__(self, params, classifier_config):
    super(CLDCClassifier, self).__init__()

    self.mlp = BaseMLP(classifier_config)

    self.criterion = nn.CrossEntropyLoss()
    # sig
    #self.criterion = nn.BCEWithLogitsLoss()
    # sig
    
    # vis
    self.vis_x = []
    self.vis_y = []
    # vis

    self.use_cuda = params.cuda


  def forward(self, x, y, training, vis = False):
    if training:
      self.train()
      pred_logits = self.mlp(x)
    else:
      self.eval()
      with torch.no_grad():
        pred_logits = self.mlp(x)

    # bs 
    pred = torch.argmax(pred_logits, dim = 1)
    # bs, label_size
    pred_p = torch.softmax(pred_logits, dim = 1)
    '''
    # sig
    pred_p = torch.sigmoid(pred_logits)
    pred = pred_p > 0.5
    pred_p = torch.cat([1 - pred_p, pred_p], dim = -1)
    # sig
    '''
    # vis
    if self.training is False and vis:
      last_hid = self.mlp.mlp[:-1](x)
      self.vis_x.append(last_hid.detach().cpu().numpy()) 
      self.vis_y.append(y.detach().cpu().numpy())
    # vis  
    
    cldc_loss = None
    if training and y is not None:
      cldc_loss = self.criterion(pred_logits, y)
      # sig
      #cldc_loss = self.criterion(pred_logits.squeeze(), y.float())
      # sig

    return cldc_loss, pred_p, pred
