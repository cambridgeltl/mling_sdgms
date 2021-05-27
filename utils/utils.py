# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
https://raw.githubusercontent.com/wohlert/semi-supervised-pytorch/master/semi-supervised/utils.py
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch

import pdb


def enumerate_discrete(x, label_size):
  """
  Generates a `torch.Tensor` of size batch_size x n_labels of
  the given label.

  Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                     [1, 0, 0]])
  :param x: tensor with batch size to mimic
  :param y_dim: number of total labels
  :return variable
  """
  y_dim = label_size 
  def batch(batch_size, label):
    labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
    y = torch.zeros((batch_size, y_dim))
    y.scatter_(1, labels, 1)
    return y.type(torch.LongTensor)

  #batch_size = x.size(0)
  #generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

  batch_size = 1
  generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])
  generated = generated.repeat(x.size(0), 1)

  return generated.float()


def onehot(label, label_size):
  k = label_size 
  y = torch.zeros(k)
  if label < k:
    y[label] = 1
  return y


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
  """
  Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
  :param tensor: Tensor to compute LSE over
  :param dim: dimension to perform operation over
  :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
  :return: LSE
  """
  max, _ = torch.max(tensor, dim=dim, keepdim=True)
  return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max
