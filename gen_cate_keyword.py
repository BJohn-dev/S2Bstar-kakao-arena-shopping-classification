# 
# gen_cate_keyword.py
# ==============================================================================
import re
import pickle
import json

import pandas as pd
import numpy as np

from konlpy.tag import Kkma

from utils_class import cate1


def splitter(sent):
    out = []
    sep_sent = sent.split(' ')
    for sen in sep_sent:
        for word in sen.split('/'):
            out.append(word)
    return out

man_cate_names = []
cateid_kr = pd.DataFrame(cate1)
for cateid in cateid_kr.index:
    for cate_name in splitter(cateid):
        man_cate_names.append(cate_name)

man_cate_names = set(man_cate_names)

# Kkma
kkma = Kkma()

cate_names = []
for cateid in cateid_kr.index:
    for cate_name in kkma.morphs(cateid):
        cate_names.append(cate_name)

cate_names = set(cate_names)

fin_cate_names = man_cate_names | set(cate_names)

should_del = ['-', '[',']','(',')','/','+','','[시리얼]']
_cate_names = []
for name in fin_cate_names:
    if name not in should_del:
        _cate_names.append(name)
fin_cate_names = _cate_names

new_cate_names = []

for name in fin_cate_names:
    if len(name) > 1:
        new_cate_names.append(name)
fin_cate_names = new_cate_names

print('Saving...')
# Save
with open('data/final_cate_names.pickle', 'wb') as f:
    pickle.dump(fin_cate_names, f)