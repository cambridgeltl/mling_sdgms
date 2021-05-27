import os

base_path = '/yourpath/data/europarl'
en_dict_path = 'europarl-v7.de-en.en.preproc.train.dict'
en_thresh = 40000
de_dict_path = 'europarl-v7.de-en.de.preproc.train.dict'
de_thresh = 50000
out_path = 'ende.word.list'

with open(os.path.join(base_path, en_dict_path), 'r') as fen, \
     open(os.path.join(base_path, de_dict_path), 'r') as fde:
  en_words = fen.readlines()
  de_words = fde.readlines()
en_words = [w.strip().split()[0].strip() for w in en_words][:en_thresh]
de_words = [w.strip().split()[0].strip() for w in de_words][:de_thresh]

merge_list = list(set(en_words + de_words))
with open(out_path, 'w') as fout:
  fout.write('\n'.join(merge_list))
