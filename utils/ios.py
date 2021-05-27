# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys


def out_parallel(i, j, n_batch, loss_batch, nll_batch, kld_batch, best_nll_dev, nll_dev, kld_dev, num_bad_epochs):
  sys.stdout.write("\rep {:2d} {:5d}/{:5d} ".format(i + 1, j + 1, n_batch) + \
      "ls={:4.2f} nll={:4.2f} kld={:4.2f} ".format(loss_batch, nll_batch, kld_batch) + \
      "bdev_nll={:4.2f} dev_nll={:4.2f} dev_kld={:4.2f} pat = {:1d}".format(best_nll_dev, nll_dev, kld_dev, num_bad_epochs))
  sys.stdout.flush()


def out_xling(i, j, n_batch, loss_batch, nll_batch, kld_batch, best_nll_dev, nll_dev, kld_dev, num_bad_epochs, ls_dis = None, ls_enc = None):
  if ls_dis is None or ls_enc is None:
    sys.stdout.write("\rep {:2d} {:5d}/{:5d} ".format(i + 1, j + 1, n_batch) + \
      "ls={:4.2f} nll={:4.2f} kld={:4.2f} ".format(loss_batch, nll_batch, kld_batch) + \
      "bdev_nll={:4.2f} dev_nll={:4.2f} dev_kld={:4.2f} ps={:1d}".format(best_nll_dev, nll_dev, kld_dev, num_bad_epochs))
  else:
    sys.stdout.write("\rep {:2d} {:5d}/{:5d} ".format(i + 1, j + 1, n_batch) + \
      "ls={:4.2f} nll={:4.2f} kld={:4.2f} ls_enc = {:4.2f} ls_dis={:4.2f} ".format(loss_batch, nll_batch, kld_batch, ls_enc, ls_dis) + \
      "bdev_nll={:4.2f} dev_nll={:4.2f} dev_kld={:4.2f} ps={:1d}".format(best_nll_dev, nll_dev, kld_dev, num_bad_epochs))
  sys.stdout.flush()


def out_cldc(i, j, n_batch, cldc_loss_batch, batch_acc, batch_acc_cls, bdev, btest, cdev, ctest, num_bad_epochs):
  sys.stdout.write("\rep {:2d} {:5d}/{:5d} ".format(i + 1, j + 1, n_batch) + \
    "ls={:5.3f} train_acc={:5.3f} ({}) ".format(cldc_loss_batch, batch_acc, ','.join(['{:5.3f}'.format(s) for s in batch_acc_cls])) + \
    "bdev={:5.3f} btest={:5.3f} ".format(bdev, btest) + \
    "cdev={:5.3f} ctest={:5.3f} ".format(cdev, ctest) + \
    "ps = {}".format(num_bad_epochs))
  sys.stdout.flush()


def out_semicldc_fixenc(i, j, n_batch, loss_dict, batch_acc, batch_acc_cls, bdev, btest, cdev, ctest, num_bad_epochs):
  sys.stdout.write("\nep {:2d} {:5d}/{:5d}\n".format(i + 1, j + 1, n_batch) +
    "cldc_ls={:5.3f} ".format(loss_dict['L_cldc_loss']) +
    "train_acc={:5.3f} ({}) ".format(batch_acc, ','.join(['{:5.3f}'.format(s) for s in batch_acc_cls])) +
    "bdev={:5.3f} btest={:5.3f} ".format(bdev, btest) +
    "cdev={:5.3f} ctest={:5.3f}\n".format(cdev, ctest) +
    "Lrec={:5.3f} Lkld={:5.3f} Lypr={:5.3f} L={:5.3f}\n".format(loss_dict['L_rec'], loss_dict['L_kld'], loss_dict['L_yprior'], loss_dict['L_loss']) +
    "Urec={:5.3f} Ukld={:5.3f} Uypr={:5.3f} U={:5.3f} ".format(loss_dict['U_rec'], loss_dict['U_kld'], loss_dict['U_yprior'], loss_dict['UL_mean_loss']) +
    "H={:5.3f} kldy={:5.3f} ".format(loss_dict['H'], loss_dict['kldy']) +
    "ps = {}".format(num_bad_epochs))
  sys.stdout.flush()


def out_semicldc_trainenc(i, j, n_batch,
                          loss_dict, batch_acc, batch_acc_cls,
                          bdev, btest, cdev, ctest, num_bad_epochs):
  sys.stdout.write("\nep {:2d} {:5d}/{:5d}\n".format(i + 1, j + 1, n_batch) +
    "cldc_ls={:5.3f} ".format(loss_dict['L_cldc_loss']) +
    "train_acc={:5.3f} ({}) ".format(batch_acc, ','.join(['{:5.3f}'.format(s) for s in batch_acc_cls])) +
    "bdev={:5.3f} btest={:5.3f} ".format(bdev, btest) +
    "cdev={:5.3f} ctest={:5.3f}\n".format(cdev, ctest) +
    "Lrec={:5.3f} Lkld={:5.3f} Lypr={:5.3f} L={:5.3f} Lnll={:5.3f} LHz1={:5.3f} Lz1kl={:5.3f} LTKL={:5.3f} Ldis={:5.3f} Lenc={:5.3f}\n".format(loss_dict['L_rec'], loss_dict['L_kld'], loss_dict['L_yprior'], loss_dict['L_loss'], loss_dict['L_nll'], loss_dict['L_Hz1'], loss_dict['L_z1kld'], loss_dict['L_TKL'], loss_dict['L_dis_loss'], loss_dict['L_enc_loss']) +
    "Urec={:5.3f} Ukld={:5.3f} Uypr={:5.3f} U={:5.3f} Unll={:5.3f} UHz1={:5.3f} Uz1kl={:5.3f} UTKL={:5.3f} Udis={:5.3f} Uenc={:5.3f} ".format(loss_dict['U_rec'], loss_dict['U_kld'], loss_dict['U_yprior'], loss_dict['UL_mean_loss'], loss_dict['U_nll'], loss_dict['U_Hz1'], loss_dict['U_z1kld'], loss_dict['U_TKL'], loss_dict['U_dis_loss'], loss_dict['U_enc_loss']) +
    "H={:5.3f} kldy={:5.3f} ".format(loss_dict['H'], loss_dict['kldy']) +
    "ps = {}".format(num_bad_epochs))
  sys.stdout.flush()


def out_xlsemicldc_trainenc(i, j, n_batch,
                            src_loss_dict, trg_loss_dict,
                            src_batch_acc, trg_batch_acc,
                            src_batch_acc_cls, trg_batch_acc_cls,
                            bdev, btest, cdev, ctest, num_bad_epochs):
  sys.stdout.write("\nep {:2d} {:5d}/{:5d}\n".format(i + 1, j + 1, n_batch) +
    "src:\n" +
    "cldc_ls={:5.3f} ".format(src_loss_dict['L_cldc_loss']) +
    "train_acc={:5.3f} ({})\n".format(src_batch_acc, ','.join(['{:5.3f}'.format(s) for s in src_batch_acc_cls])) +
    "Lrec={:5.3f} Lkld={:5.3f} Lypr={:5.3f} L={:5.3f} Lnll={:5.3f} LHz1={:5.3f} Lz1kl={:5.3f} LTKL={:5.3f} Ldis={:5.3f} Lenc={:5.3f}\n".format(
    src_loss_dict['L_rec'], src_loss_dict['L_kld'], src_loss_dict['L_yprior'],
    src_loss_dict['L_loss'], src_loss_dict['L_nll'], src_loss_dict['L_Hz1'],
    src_loss_dict['L_z1kld'], src_loss_dict['L_TKL'],
    src_loss_dict['L_dis_loss'], src_loss_dict['L_enc_loss']) +
    "Urec={:5.3f} Ukld={:5.3f} Uypr={:5.3f} U={:5.3f} Unll={:5.3f} UHz1={:5.3f} Uz1kl={:5.3f} UTKL={:5.3f} Udis={:5.3f} Uenc={:5.3f} ".format(
    src_loss_dict['U_rec'], src_loss_dict['U_kld'], src_loss_dict['U_yprior'],
    src_loss_dict['UL_mean_loss'], src_loss_dict['U_nll'], src_loss_dict['U_Hz1'],
    src_loss_dict['U_z1kld'], src_loss_dict['U_TKL'],
    src_loss_dict['U_dis_loss'], src_loss_dict['U_enc_loss']) +
    "H={:5.3f} kldy={:5.3f} ".format(src_loss_dict['H'], src_loss_dict['kldy']) +

    "\ntrg:\n" +
    "cldc_ls={:5.3f} ".format(trg_loss_dict['L_cldc_loss']) +
    "train_acc={:5.3f} ({})\n".format(trg_batch_acc, ','.join(['{:5.3f}'.format(s) for s in trg_batch_acc_cls])) +
    "Lrec={:5.3f} Lkld={:5.3f} Lypr={:5.3f} L={:5.3f} Lnll={:5.3f} LHz1={:5.3f} Lz1kl={:5.3f} LTKL={:5.3f} Ldis={:5.3f} Lenc={:5.3f}\n".format(
    trg_loss_dict['L_rec'], trg_loss_dict['L_kld'], trg_loss_dict['L_yprior'],
    trg_loss_dict['L_loss'], trg_loss_dict['L_nll'], trg_loss_dict['L_Hz1'],
    trg_loss_dict['L_z1kld'], trg_loss_dict['L_TKL'],
    trg_loss_dict['L_dis_loss'], trg_loss_dict['L_enc_loss']) +
    "Urec={:5.3f} Ukld={:5.3f} Uypr={:5.3f} U={:5.3f} Unll={:5.3f} UHz1={:5.3f} Uz1kl={:5.3f} UTKL={:5.3f} Udis={:5.3f} Uenc={:5.3f} ".format(
    trg_loss_dict['U_rec'], trg_loss_dict['U_kld'], trg_loss_dict['U_yprior'],
    trg_loss_dict['UL_mean_loss'], trg_loss_dict['U_nll'], trg_loss_dict['U_Hz1'],
    trg_loss_dict['U_z1kld'], trg_loss_dict['U_TKL'],
    trg_loss_dict['U_dis_loss'], trg_loss_dict['U_enc_loss']) +
    "H={:5.3f} kldy={:5.3f} ".format(trg_loss_dict['H'], trg_loss_dict['kldy']) +
    "\nbdev={:5.3f} btest={:5.3f} ".format(bdev, btest) +
    "cdev={:5.3f} ctest={:5.3f} ".format(cdev, ctest) +
    "ps = {}".format(num_bad_epochs))
  sys.stdout.flush()
