# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import threading
import operator
import pickle

import fire
import h5py
import tqdm
import numpy as np
import six

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from six.moves import zip, cPickle

from misc import get_logger, Option
from network import TextOnly, top1_acc, customLoss

opt = Option('./config.json')
if six.PY2:
    cate1 = json.loads(open('../cate1.json').read())
else:
    cate1 = json.loads(open('../cate1.json', 'rb').read().decode('utf-8'))


import glob
import pandas as pd


import time
from contextlib import contextmanager
from functools import partial
@contextmanager
def timer(title):
    print("###### Start {}".format(title))
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

    
def dict_max(dictionary):
    _a = max(dictionary.items(), key=operator.itemgetter(1))[0]
    if _a == -1:
        try: return sorted(dictionary.items(), key=lambda kv: kv[1])[-2][0]
        except: return _a
    return _a

def post_process_first_stage(line):

    cate_dict = {}
    for i in range(1,5):
        cate_dict[i] = {}
        for rank in range(5):  
                try: cate_dict[i][line[5*rank+i+1]] += line[5*rank+1]
                except: cate_dict[i][line[5*rank+i+1]] = line[5*rank+1]

    new_line = [line[0], dict_max(cate_dict[1]), dict_max(cate_dict[2]), dict_max(cate_dict[3]), dict_max(cate_dict[4])]

    for i in range(1, 5):
        for j in range(2, 6):
            new_line.append(line[i*5+j])

    return new_line

pds_df = pickle.load(open('post_processing_model/ml_pds_data.pickle','rb'))
psm_df = pickle.load(open('post_processing_model/ml_psm_data.pickle','rb'))

pds_list = pds_df[['bcateid', 'mcateid', 'scateid']].values.tolist()

def ml_changer(line, max_rank=5):
    for rank in range(max_rank):
        if line[3+rank*4] == -1.0:
            if line[1+rank*4:3+rank*4] in psm_df[['bcateid', 'mcateid']].values.tolist():
                line[3+rank*4] = psm_df.loc[(psm_df['bcateid']==line[1+rank*4])&(psm_df['mcateid']==line[2+rank*4])]['scateid'].tolist()[0]
        if line[3+rank*4] != -1.0 and line[4+rank*4] == -1.0:
            if line[1+rank*4:4+rank*4] in pds_df[['bcateid', 'mcateid', 'scateid']].values.tolist():
                line[4+rank*4] = pds_df.loc[(pds_df['bcateid']==line[1+rank*4])&(pds_df['mcateid']==line[2+rank*4])&(pds_df['scateid']==line[3+rank*4])]['dcateid'].tolist()[0]
    return line

def rank_puller(line, max_rank=3):
    if line[3]==-1:
        for rank in range(1,max_rank):
            if line[3+rank*4] != -1:
                line[3] = line[3+rank*4]
                break
    if line[4]==-1:
        for rank in range(1,max_rank):
            if line[4+rank*4] != -1:
                line[4] = line[4+rank*4]
                break
    return line[0:5]

b, d, _r_scateid, _r_dcateid = pickle.load(open('post_processing_model/tools_map.pkl', 'rb'))
def get_new_s_pre(b_pre, m_pre, s_pre):
    if s_pre == -1:
        try:
            return b.loc[(b_pre, m_pre)].index.values[0]
        except:
            return s_pre
        
    return s_pre
        
def get_new_d_pre(b_pre, m_pre, d_pre):
    if d_pre == -1:
        try:
            return d.loc[(b_pre, m_pre)].index.values[0]
        except:
            return d_pre
        
    return d_pre


def get_new_s_pre_only_b_pre(b_pre, s_pre):
    if s_pre == -1:
        try:
            return _r_scateid.loc[b_pre].index.values[0]
        except:
            return s_pre
        
    return s_pre
        
def get_new_d_pre_only_b_pre(b_pre, d_pre):
    if d_pre == -1:
        try:
            return _r_dcateid.loc[b_pre].index.values[0]
        except:
            return d_pre
        
    return d_pre


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0

    def get_sample_generator(self, ds, batch_size, raise_stop_event=False):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right] for t in ['embd_word', 'img_feat']]
            
            Y = ds['cate'][left:right]
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in six.iteritems(cate1[d])}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, pred_y_top_n, confid_top_n, meta, out_path, dataset_name):
        
        DEV_DATA_LIST = None
        if dataset_name == 'dev':
            DEV_DATA_LIST = ['../dev.chunk.01']
        elif dataset_name == 'test':
            DEV_DATA_LIST = ['../test.chunk.01', '../test.chunk.02']
            
        print("$$$$$$", DEV_DATA_LIST)
        pid_order = []
        for data_path in DEV_DATA_LIST:
            h = h5py.File(data_path, 'r')[dataset_name]
            pid_order.extend(h['pid'][::])


        y2l = {i: s for s, i in six.iteritems(meta['y_vocab'])}
        y2l = list(map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0])))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}

        pid2label = {}
        for pid, y, y_top_n, p_top_n in tqdm.tqdm(zip(data['pid'], pred_y, pred_y_top_n, confid_top_n)):
            if six.PY3:
                pid = pid.decode('utf-8')
            aaa = "{}".format(pid)
            aaa_list = [pid]

            for i, j in zip(y_top_n, p_top_n):
                ll = y2l[i] # Lower case of LL
                tkns = list(map(int, ll.split('>')))
                aaa += "\t{}\t{}\t{}\t{}\t{}".format(j, tkns[0], tkns[1], tkns[2], tkns[3])
                aaa_list.extend([j, tkns[0], tkns[1], tkns[2], tkns[3]])
            
            #################################################################################
            #################### MAP
            
            aaa_list_for_map = aaa_list[:]
            
            aaa_list_for_map[4] = get_new_s_pre(aaa_list[2], aaa_list[3], aaa_list[4])
            aaa_list_for_map[5] = get_new_d_pre(aaa_list[2], aaa_list[3], aaa_list[5])

            aaa_list_for_map[4] = get_new_s_pre_only_b_pre(aaa_list[2], aaa_list[4])
            aaa_list_for_map[5] = get_new_d_pre_only_b_pre(aaa_list[2], aaa_list[5])
            
            
            aaa_list_for_map = [aaa_list_for_map[i] for i in [0, 2, 3, 4, 5]]
            
            # print("@#@#", aaa_list_for_map)
            #################################################################################

            #################################################################################
            #################### MLE
            
            # print(aaa_list)
            aaa_list = post_process_first_stage(aaa_list)
            # print(aaa_list)
            # print(ml_changer(aaa_list))
            aaa_list = ml_changer(aaa_list)
            pre_processed_list = rank_puller(aaa_list, max_rank=5)
            # print(pre_processed_list)
            
            #################################################################################
            
            #################################################################################
            #################### MLE + MAP
            
            
            for index in range(1, len(pre_processed_list)):
                
                if int(pre_processed_list[index]) == -1:
                    pre_processed_list[index] = int(aaa_list_for_map[index])
                    
            # print(pre_processed_list)
            #################################################################################

            pid2label[pid] = "\t".join(list(map(str, pre_processed_list)))
            # print(pid2label[pid])
                                       
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        kkkk = 0 
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
                ans = pid2label.get(pid, no_answer.format(pid=pid))
                
                if ans == no_answer.format(pid=pid):
                    kkkk+=1
                fout.write(ans)
                fout.write('\n')
            # print("@#$@!#$!@#$@!#$@!#$@#!$ ", kkkk)

    def predict(self, dataset_name):
        
        #################################################
        #################################################        
            
        test_root = '../dataset_word2vec_price2vec_img_feat_updttm_new/data/{}/'.format(dataset_name)
        model_root = 'model/train/'

        test_div = 'dev'
        
        out_path = 'output/retrain.light.post_processing.predict.{}.tsv'.format(dataset_name)
        
        meta_path = os.path.join(test_root, 'meta')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        model_fname = os.path.join(model_root, 'weights')
        self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        model = load_model(model_fname,
                           custom_objects={'top1_acc': top1_acc, 'customLoss': customLoss})

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size
        pred_y = []
        pred_y_top_n = []
        confid_top_n = []
        n = 5
        test_gen = ThreadsafeIter(self.get_sample_generator(test, batch_size, raise_stop_event=True))
        total_test_samples = test['uni'].shape[0]
        
#         jjjj = 0
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['uni'].shape[0]
                X, _ = chunk
                _pred_y = model.predict(X)
                pred_y.extend([np.argmax(y) for y in _pred_y])
                
#                 y = _pred_y[0]
#                 print("###", 
#                       type(_pred_y), 
#                       np.shape(_pred_y), 
#                       type(y), 
#                       np.shape(y), 
#                       type(np.argsort(y)[-n:][::-1]), 
#                       np.shape(np.argsort(y)[-n:][::-1]))
    
                pred_y_top_n.extend([np.argsort(y)[-n:][::-1] for y in _pred_y])
                confid_top_n.extend([y[np.argsort(y)[-n:][::-1]] for y in _pred_y])
                
#                 if jjjj > 1:
#                     print("\n$$$$$$$$$$$$$", np.shape(_pred_y))
#                     print("\n$$$$$$$$$$$$$", np.shape([np.argsort(y)[-n:][::-1] for y in _pred_y]))
#                     print("\n$$$$$$$$$$$$$", np.shape(pred_y_top_n))
#                     print("\n$$$$$$$$$$$$$", np.shape(confid_top_n))
#                     break
                
#                 jjjj+=1
                
                pbar.update(X[0].shape[0])
        self.write_prediction_result(test, pred_y, pred_y_top_n, confid_top_n, meta, out_path, dataset_name)


class ThreadsafeIter(object):
    def __init__(self, it):
        self._it = it
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._it)

    def next(self):
        with self._lock:
            return self._it.next()

if __name__ == '__main__':
    dataset_name = sys.argv[1]    
    clsf = Classifier()
    clsf.predict(dataset_name)
