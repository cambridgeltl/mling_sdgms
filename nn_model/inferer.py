# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Inferer from h to z
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np

import torch
import torch.nn as nn

import pdb


class Inferer(nn.Module):
  def __init__(self, params, in_dim):
    super(Inferer, self).__init__()
    self.in_dim = in_dim
    self.hid_dim = params.z_dim 
    self.i2h = nn.Linear(self.in_dim, self.hid_dim)
    self.hbn = nn.BatchNorm1d(self.hid_dim)
    self.relu = nn.ReLU()

    self.mu = nn.Linear(self.hid_dim, self.hid_dim)
    self.logvar = nn.Linear(self.hid_dim, self.hid_dim)

    self.use_cuda = params.cuda


  def forward(self, hids):
    in_emb = hids

    hid = self.i2h(in_emb)
    # avoid only one instance for BN
    if hids.shape[0] > 1:
      hid = self.hbn(hid)
    hid = self.relu(hid)

    mu = self.mu(hid)
    logvar = self.logvar(hid)
    return mu, logvar


  def reparameterize(self, mu, logvar):
    if self.training:
      # std = squrae root of exp(logvar)
      std = torch.exp(0.5 * logvar)
      # sample from (0, 1) normal
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      # During inference, we simply spit out the mean of the
      # learned distribution for the current input.  We could
      # use a random sample from the distribution, but mu of
      # course has the highest probability.
      return mu
