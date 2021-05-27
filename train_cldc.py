# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
training CLDC
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import math
import random
from random import shuffle
import torch
import torch.optim as optim
import numpy as np
np.set_printoptions(precision = 4)

from data_model.cldc_data_reader import CLDCDataReader
from nn_model.cldc_model import CLDCModel
from utils.early_stopping import EarlyStopping
from utils.ios import out_xling, out_cldc
from tensorboardX import SummaryWriter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import pdb


def main(params, vocabs, model_dict):
  # get data 
  datas = get_data(params, vocabs)
  # get model
  m = CLDCModel(params, datas, params.cldc_classifier_config, model_dict = model_dict) 
  gen_task_info(params, m, datas)
  train(params, m, datas)
  return datas


def get_data(params, vocabs):
  datas = []
  # CLDC task
  for lang_idx, lang in enumerate(params.cldc_data_langs):
    if params.seg_type == 'word':
      assert(lang == vocabs[params.lang_dict[lang]].lang)
      datas.append(CLDCDataReader(params, lang_idx, vocabs[params.lang_dict[lang]]))
    elif params.seg_type == 'spm':
      datas.append(CLDCDataReader(params, lang_idx, vocabs[0]))
  return datas


def train(params, m, datas):
  # early stopping
  es = EarlyStopping(mode = 'max', patience = params.cldc_patience)
  # set optimizer
  optimizer = get_optimizer(params, m)

  # training on one lang, and dev/test for another lang
  # get training
  train_lang, train_data = get_lang_data(params, datas, training = True)
  # get dev and test, dev is the same language as test
  test_lang, test_data = get_lang_data(params, datas)

  n_batch = train_data.train_size // params.cldc_bs if train_data.train_size % params.cldc_bs == 0 else train_data.train_size // params.cldc_bs + 1
  # per category
  data_idxs = [list(range(len(train_idx))) for train_idx in train_data.train_idxs]
 
  # number of iterations
  cur_it = 0
  # write to tensorboard
  writer = SummaryWriter('./history/{}'.format(params.log_path)) if params.write_tfboard else None
  # best xx 
  bdev = 0
  btest = 0
  # current xx
  cdev = 0
  ctest = 0
  dev_class_acc = {}
  test_class_acc = {}
  dev_cm = None
  test_cm = None
  # early stopping warm up flag, start es after some iters
  es_flag = False

  for i in range(params.cldc_ep):
    for data_idx in data_idxs:
      shuffle(data_idx)
    for j in range(n_batch):
      train_idxs = []
      for k, data_idx in enumerate(data_idxs):
        if j < n_batch - 1:
          train_idxs.append(data_idx[int(j * params.cldc_bs * train_data.train_prop[k]): int((j + 1) * params.cldc_bs * train_data.train_prop[k])])
        elif j == n_batch - 1:
          train_idxs.append(data_idx[int(j * params.cldc_bs * train_data.train_prop[k]):])

      batch_train, batch_train_lens, batch_train_lb = get_batch(params, train_idxs, train_data.train_idxs, train_data.train_lens)
      optimizer.zero_grad()
      m.train()

      cldc_loss_batch, _, batch_pred = m(train_lang, batch_train, batch_train_lens, batch_train_lb)

      batch_acc, batch_acc_cls = get_classification_report(params, batch_train_lb.data.cpu().numpy(), batch_pred.data.cpu().numpy())

      if cldc_loss_batch < params.cldc_lossth:
        es_flag = True

      cldc_loss_batch.backward()
      out_cldc(i, j, n_batch, cldc_loss_batch, batch_acc, batch_acc_cls, bdev, btest, cdev, ctest, es.num_bad_epochs)

      optimizer.step()
      cur_it += 1
      update_tensorboard(writer, cldc_loss_batch, batch_acc, cdev, ctest, dev_class_acc, test_class_acc, cur_it)
      
      if cur_it % params.CLDC_VAL_EVERY == 0:
        sys.stdout.write('\n') 
        sys.stdout.flush()
        # validation 
        #cdev, dev_class_acc, dev_cm = test(params, m, test_data.dev_idxs, test_data.dev_lens, test_data.dev_size, test_data.dev_prop, test_lang, cm = True)
        cdev, dev_class_acc, dev_cm = test(params, m, train_data.dev_idxs, train_data.dev_lens, train_data.dev_size, train_data.dev_prop, train_lang, cm = True)
        ctest, test_class_acc, test_cm = test(params, m, test_data.test_idxs, test_data.test_lens, test_data.test_size, test_data.test_prop, test_lang, cm = True)
        print(dev_cm)
        print(test_cm)
        if es.step(cdev):
          print('\nEarly Stoped.')
          return
        elif es.is_better(cdev, bdev):
          bdev = cdev
          btest = ctest
          #save_model(params, m)
        # reset bad epochs
        if not es_flag:
          es.num_bad_epochs = 0


def test(params, m, input_idxs, input_lens, input_size, input_prop, input_lang, cm = False, vis = False):
  m.eval() 
  n_batch = input_size // params.cldc_test_bs if input_size % params.cldc_test_bs == 0 else input_size // params.cldc_test_bs + 1
  data_idxs = [list(range(len(input_idx))) for input_idx in input_idxs]
  acc = .0
  labels = list(params.cldc_label2idx.keys())

  preds = []
  sorted_y = []

  for j in range(n_batch):
    test_idxs = []
    for k, data_idx in enumerate(data_idxs):
      if j < n_batch - 1:
        test_idxs.append(data_idx[int(j * params.cldc_test_bs * input_prop[k]): int((j + 1) * params.cldc_test_bs * input_prop[k])])
      elif j == n_batch - 1:
        test_idxs.append(data_idx[int(j * params.cldc_test_bs * input_prop[k]):])

    with torch.no_grad():
      batch_test, batch_test_lens, batch_test_lb = get_batch(params, test_idxs, input_idxs, input_lens)
      if params.task == 'cldc':
        _, _, batch_pred = m(input_lang, batch_test, batch_test_lens, batch_test_lb, vis = vis)
      elif 'semi' in params.task:
        _, _, batch_pred = m.train_classifier(input_lang, batch_test, batch_test_lens, batch_test_lb, training = False)

    acc += accuracy_score(batch_test_lb.data.cpu().numpy(), batch_pred.data.cpu().numpy(), normalize = False)
    preds += batch_pred.data.cpu().tolist()
    sorted_y += batch_test_lb.data.cpu().tolist()

  acc /= input_size
  _, recall, _, support = precision_recall_fscore_support(np.array(sorted_y), np.array(preds))
  recall = dict(zip(labels, recall))

  if cm is True:
    cm = get_cm(np.array(sorted_y), np.array(preds))
    return acc, recall, cm
  else:
    return acc, recall


def get_optimizer(params, m):
  # set optimizer
  if params.cldc_train_mode == 'fixenc':
    ps = [p[1] for p in m.named_parameters() if 'cldc_classifier' in p[0]]
  elif params.cldc_train_mode == 'trainenc':
    ps = list(filter(lambda p: p.requires_grad, m.parameters()))
  print('Model parameter: {}'.format(sum(p.numel() for p in ps)))
  optimizer = optim.Adam(ps, lr = params.cldc_init_lr)
  return optimizer


def get_batch(params, idxs, input_idxs, input_lens):
  batch_x = []
  batch_x_lens = []
  batch_y = []
  for i, idx in enumerate(idxs):
    # per category
    if idx:
      batch_x.append(input_idxs[i][idx])
      batch_x_lens.append(input_lens[i][idx])
      batch_y.append([i] * len(idx))
  batch_x = np.concatenate(batch_x)
  batch_x_lens = np.concatenate(batch_x_lens)
  batch_y = np.concatenate(batch_y)

  # sort in the descending order
  sorted_len_idxs = np.argsort(-batch_x_lens)
  sorted_batch_x_lens = batch_x_lens[sorted_len_idxs]
  sorted_batch_x = batch_x[sorted_len_idxs]
  sorted_batch_x = torch.LongTensor(sorted_batch_x)
  sorted_batch_y = batch_y[sorted_len_idxs]
  sorted_batch_y = torch.LongTensor(sorted_batch_y)

  if params.cuda:
    sorted_batch_x = sorted_batch_x.cuda()
    sorted_batch_y = sorted_batch_y.cuda()

  return sorted_batch_x, sorted_batch_x_lens, sorted_batch_y


def get_lang_data(params, datas, training = False):
  # 0 for training, 1 for test
  lang = params.cldc_langs[0] if training else params.cldc_langs[1]
  data = datas[params.cldc_data_lang_dict[lang]]
  return lang, data


def gen_task_info(params, m, datas):
  gen_task_gen_info(params, m, datas)
  # cldc sepcific info
  cldc_spec_log = ('Init lr: {}\n'.format(params.cldc_init_lr) + 
                   '{}'.format('=' * 80)
                  )
  print(cldc_spec_log)


def gen_task_gen_info(params, m, datas):
  train_lang, train_data = get_lang_data(params, datas, training = True)
  # get dev and test, dev is the same language as test
  test_lang, test_data = get_lang_data(params, datas)
  if params.task == 'xl-semi-cldc' or params.task == 'aux-xl-semi-cldc':
    cldc_train_scale = '{} {}'.format(params.cldc_train_scale, params.cldc_xl_train_scale)
    train_sizes = '{} [{}] {}; {} [{}] {}'.format(train_data.train_size, 
                                                   ' '.join([str(len(idx)) for idx in train_data.train_idxs]), 
                                                   train_data.rest_train_size,
                                                   test_data.train_size, 
                                                   ' '.join([str(len(idx)) for idx in test_data.train_idxs]), 
                                                   test_data.rest_train_size)
    # max lens for train/dev/test
    max_lens = list(map(str, [train_data.max_train_len, test_data.max_train_len, test_data.max_dev_len, test_data.max_test_len]))
    max_lens = '[{}]'.format(' '.join(max_lens))
  else:
    cldc_train_scale = params.cldc_train_scale
    train_sizes = '{} [{}] {}'.format(train_data.train_size, 
                                        ' '.join([str(len(idx)) for idx in train_data.train_idxs]), 
                                        train_data.rest_train_size)
    # max lens for train/dev/test
    max_lens = list(map(str, [train_data.max_train_len, test_data.max_dev_len, test_data.max_test_len]))
    max_lens = '[{}]'.format(' '.join(max_lens))

  # general info for cldc
  cldc_log = ('Train lang: {}, dev/test lang: {}\n'.format(train_lang, test_lang) + 
              'Label size: {}\n'.format(params.cldc_label_size) + 
              'Labels: [{}]\n'.format(' '.join([lb for idx, lb in params.cldc_idx2label.items()])) + 
              'Train percentage: {}\n'.format(cldc_train_scale) + 
              'Val every: {}\n'.format(params.CLDC_VAL_EVERY) + 
              'Train size: {}\n'.format(train_sizes) + 
              'Dev size: {} [{}]\n'.format(test_data.dev_size, ' '.join([str(len(idx)) for idx in test_data.dev_idxs])) + 
              'Test size: {} [{}]\n'.format(test_data.test_size, ' '.join([str(len(idx)) for idx in test_data.test_idxs])) +
              'Sentence lenper: {}\n'.format(max_lens) +
              'Train encoder: {}\n'.format(params.cldc_train_mode) +
              'q(y|x): [{}]\n'.format(' '.join(list(map(str, params.cldc_classifier_config)))) +
              'Early stopping: {} {}\n'.format(params.cldc_patience, params.cldc_lossth) +
              'Batch size: {} {}'.format(params.cldc_bs, params.cldc_test_bs)
      )
  print(cldc_log)


def tsne2d(params, m):
    y_names, y_idxs = params.cldc_label2idx.keys(), params.cldc_label2idx.values()
    vis_x = np.concatenate(m.cldc_classifier.vis_x)
    vis_y = np.concatenate(m.cldc_classifier.vis_y)
    # project to 2d
    tsne = TSNE(n_components = 2, random_state = 0)
    vis_x = tsne.fit_transform(vis_x)
    colors = 'r', 'g', 'b', 'y'
    plt.figure(figsize = (20, 20))

    for idx, c, label in zip(y_idxs, colors, y_names):
      plt.scatter(vis_x[vis_y == idx, 0],
                  vis_x[vis_y == idx, 1],
                  c = c, label = label)
    plt.savefig('{}.{}.jpg'.format(params.task, len(vis_x)))
    plt.clf()



def get_classification_report(params, y_true, y_pred):
  score = classification_report(y_true, y_pred, output_dict = True)
  acc = score['accuracy']
  acc_cls = [score[str(s)]['recall'] if str(s) in score else .0 for s in params.cldc_idx2label.keys()]
  return acc, acc_cls


def get_cm(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  return cm


def save_model(params, m):
  print('Save the model ...')
  model_dict = m.state_dict()
  model_dict = {k: v.cpu() for k, v in model_dict.items()}
  torch.save(model_dict, '{}.pth'.format(params.log_path))


def update_tensorboard(writer, cldc_loss_batch, cldc_acc_batch, cdev, ctest, dev_class_acc, test_class_acc, cur_it):
  if writer is None:
    return
  writer.add_scalar("train/loss", cldc_loss_batch, cur_it)
  writer.add_scalar("train/acc", cldc_acc_batch, cur_it)
  writer.add_scalar("dev/cur_acc", cdev, cur_it)
  writer.add_scalar("test/cur_acc", ctest, cur_it)
  if dev_class_acc and test_class_acc:
    writer.add_scalars("dev/class_acc", dev_class_acc, cur_it)
    writer.add_scalars("test/class_acc", test_class_acc, cur_it)
