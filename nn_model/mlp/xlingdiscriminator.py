# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Language Discriminator
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.nn as nn

from .base_mlp import BaseMLP

import pdb


class XlingDiscriminator(nn.Module):
  def __init__(self, params):
    super(XlingDiscriminator, self).__init__()
    
    self.mlp = BaseMLP(params.xlingdiscriminator_config)
    self.criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    # label = 1 of discriminator
    self.dis_lang = params.langs[0]

    self.use_cuda = params.cuda


  def forward(self, lang, hid):
    o = self.mlp(hid)

    # 1 for dis_lang
    dis_label = torch.ones_like(o) if lang == self.dis_lang else torch.zeros_like(o)
    enc_label = 1 - dis_label
    loss_dis = self.criterion(o, dis_label)
    loss_enc = self.criterion(o, enc_label)

    return loss_dis, loss_enc
