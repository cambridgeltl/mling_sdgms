from scipy.sparse import csc_matrix
import os
import _pickle as cPickle
import re

import pdb

dic = {}

de_corpus = []
en_corpus = []

de_dict = {}
en_dict = {}

# include UNK
de_voc_size = 50001
en_voc_size = 40001

print('calculating word frequency')

with open('/yourpath/data/europarl/europarl-v7.de-en.en.preproc.train', 'r') as fin_en, open('/yourpath/data/europarl/europarl-v7.de-en.de.preproc.train', 'r') as fin_de:
  en_str = fin_en.readlines()
  de_str = fin_de.readlines()
  assert(len(en_str) == len(de_str))

for line_id in range(len(en_str)):
  de_sent = de_str[line_id].strip().split(' ')
  de_sent = [w for w in de_sent if w]
  en_sent = en_str[line_id].strip().split(' ')
  en_sent = [w for w in en_sent if w]
  
  for token in de_sent:
    if token not in de_dict:
      de_dict[token] = 1
    else:
      de_dict[token] += 1
  for token in en_sent:
    if token not in en_dict:
      en_dict[token] = 1
    else:
      en_dict[token] += 1

if '' in de_dict:
  print('delete empty string')
  del de_dict['']
if '' in en_dict:
  print('delete empty string')
  del en_dict['']
print('original de voc size:', len(de_dict))
print('original en voc size:', len(en_dict))

s_de = sorted(de_dict.items(), key=lambda x: x[1], reverse=True)
s_de.insert(0, ('<UNK>', -1))
s_en = sorted(en_dict.items(), key=lambda x: x[1], reverse=True)
s_en.insert(0, ('<UNK>', -1))

de_voc_size = min(de_voc_size, len(de_dict))
en_voc_size = min(en_voc_size, len(en_dict))

print('now de dice size:', de_voc_size)
print('now en dice size:', en_voc_size)

print('writing dict to file')
# save word dict for
f_dict_de = open("de_dict.txt", 'w')
f_dict_en = open("en_dict.txt", 'w')

for i in range(de_voc_size):
    f_dict_de.write(s_de[i][0] + '\t' + str(i) + '\t' + str(s_de[i][1]) + '\n')
    de_dict[s_de[i][0]] = i
f_dict_de.close()

for i in range(en_voc_size):
    f_dict_en.write(s_en[i][0] + '\t' + str(i) + '\t' + str(s_en[i][1]) + '\n')
    en_dict[s_en[i][0]] = i
f_dict_en.close()

print('creating sparse format of document representation')

de_row = []
de_col = []
de_data = []
en_row = []
en_col = []
en_data = []

lstm_en_f = open('en.seq', 'w')
lstm_de_f = open('de.seq', 'w')

de_seq = []
en_seq = []

de_one_doc = []
en_one_doc = []
# count number of words
de_d = []
# row id
de_r = []
# col id
de_c = []
en_d = []
en_r = []
en_c = []

with open('../../data/europarl/europarl-v7.de-en.en.preproc', 'r') as fin_en, open('../../data/europarl/europarl-v7.de-en.de.preproc', 'r') as fin_de:
  en_str = fin_en.readlines()
  de_str = fin_de.readlines()
  
for line_id in range(len(en_str)):
  de_sent = de_str[line_id].strip().split(' ')
  de_sent = [w for w in de_sent if w]
  en_sent = en_str[line_id].strip().split(' ')
  en_sent = [w for w in en_sent if w]

  for token in de_sent:
    de_r.append(line_id)
    de_c.append(de_dict[token] if token in de_dict else de_dict['<UNK>'])
    de_d.append(1)
    de_one_doc.append(de_dict[token] if token in de_dict else de_dict['<UNK>'])

  for token in en_sent:
    en_r.append(line_id)
    en_c.append(en_dict[token] if token in en_dict else en_dict['<UNK>'])
    en_d.append(1)
    en_one_doc.append(en_dict[token] if token in en_dict else en_dict['<UNK>'])

  de_row += de_r
  de_col += de_c
  de_data += de_d
  en_row += en_r
  en_col += en_c
  en_data += en_d
  en_seq.append(en_one_doc)
  de_seq.append(de_one_doc)
  de_one_doc = []
  en_one_doc = []
  de_d = []
  de_r = []
  de_c = []
  en_d = []
  en_r = []
  en_c = []

assert(len(de_seq) == len(en_seq))
print('seq doc length', len(de_seq), len(en_seq))

max_de = 0
cnt = 0
for j in range(len(de_seq)):
  if max_de < len(de_seq[j]):
    max_de = len(de_seq[j])
  if len(de_seq[j]) == 0:
    print("error", j)
  lstm_de_f.write(str(de_seq[j][0]))
  for i in range(1, len(de_seq[j])):
    lstm_de_f.write("\t" + str(de_seq[j][i]))
  lstm_de_f.write("\n")
lstm_de_f.close()

max_en = 0
for j in range(len(en_seq)):
  if max_en < len(en_seq[j]):
    max_en = len(en_seq[j])

  lstm_en_f.write(str(en_seq[j][0]))
  for i in range(1, len(en_seq[j])):
    lstm_en_f.write("\t" + str(en_seq[j][i]))
  lstm_en_f.write("\n")
lstm_en_f.close()

print("max de", max_de, "max en", max_en)

de = csc_matrix((de_data, (de_row, de_col)), shape=(len(de_seq), de_voc_size))
en = csc_matrix((en_data, (en_row, en_col)), shape=(len(en_seq), en_voc_size))

print(de_voc_size, en_voc_size)
with open('en.pkl', 'wb') as pickle_file:
  cPickle.dump(en, pickle_file, protocol = 2)

with open('de.pkl', 'wb') as pickle_file:
  cPickle.dump(de, pickle_file, protocol = 2)
