# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Base class of MLP
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.nn as nn

import pdb


class BaseMLP(nn.Module):
  def __init__(self, config):
    super(BaseMLP, self).__init__()
    """
      configs are MLP configs
      int: hidden layer dimensions
      dropout_rate: dropout and its rate for the layer
      activateions: different activation functions
      config = [ops, in_dim, h1_dim, ops, h2_dim, ..., h3_dim ..., out_dim]
      #TODO: to add other settings
      * batchnorm1d cannot be used in sequential, as it can throw error when bs = 1
    """
    
    module_list = []

    i = 0 
    while i < len(config):
      if type(config[i]) == int:
        # i is in_dim, i+1 is out_dim
        module_list.append(nn.Linear(config[i], config[i + 1]))
        i += 1
      elif config[i].startswith('dropout'):
        # dropout
        do_rate = float(config[i].split('_')[-1])
        module_list.append(nn.Dropout(p = do_rate))
      elif config[i] == 'tanh':
        # tanh activations
        module_list.append(nn.Tanh())
      elif config[i] == 'sigmoid':
        # tanh activations
        module_list.append(nn.Sigmoid())
      elif config[i] == 'relu':
        # tanh activations
        module_list.append(nn.ReLU())
      elif config[i] == 'leakyrelu':
        # tanh activations
        module_list.append(nn.LeakyReLU())
      i += 1
    
    # build mlp
    self.mlp = nn.Sequential(*module_list)


  def forward(self, batch_in):
    return self.mlp(batch_in)
