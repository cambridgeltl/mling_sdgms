# -*- coding: UTF-8 -*-
#!/usr/bin/python3

#************************************************************
# Imported Libraries
#************************************************************
import sympy
import torch
import numpy as np

import pdb

const = (- 0.5 * sympy.log(2 * sympy.pi)).evalf(64)

def multi_diag_normal(x, mean, logvar):
  # bs, d
  logpx = (np.float128(const) - 0.5 * logvar - (x - mean) ** 2 / (2 * torch.exp(logvar))).sum(dim = -1)
  '''
  # pytorch multivariate gaussian
  logpx = []
  for i in range(x.shape[0]):
    m = torch.distributions.MultivariateNormal(mean[i], torch.diag(torch.exp(logvar[i])))
    logpx.append(m.log_prob(x[i]))
  logpx = torch.stack(logpx)
  '''
  return logpx


def cal_kl_gau1(mu, logvar):
  # KL(N(mu, logvar) || N(0, I))
  kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
  return kld


def cal_kl_gau2(mu_q, logvar_q, mu_p, logvar_p):
  # KL(q || p)
  kld = 0.5 * torch.sum(logvar_p - logvar_q - 1 +
        ((mu_q - mu_p).pow(2) + logvar_q.exp()) / logvar_p.exp(), dim=1)
  return kld


def cal_kl_gau1_fb(mu, logvar, fb):
  # sum(max(fb, KL(N(mu, logvar) || N(0, I)))
  dim_fb = fb * 1.0 / mu.shape[1]
  fb_kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
  kl_mask = (fb_kld > dim_fb).float()
  fb_kld = torch.sum(fb_kld * kl_mask, dim = -1)
  return fb_kld


def cal_kl_gau2_fb(mu_q, logvar_q, mu_p, logvar_p, fb):
  # sum(max(fb, KL(q || p))
  dim_fb = fb * 1.0 / mu_q.shape[1]
  fb_kld = 0.5 * (logvar_p - logvar_q - 1 + ((mu_q - mu_p).pow(2) + logvar_q.exp()) / logvar_p.exp())
  kl_mask = (fb_kld > dim_fb).float()
  fb_kld = torch.sum(fb_kld * kl_mask, dim = -1)
  return fb_kld
