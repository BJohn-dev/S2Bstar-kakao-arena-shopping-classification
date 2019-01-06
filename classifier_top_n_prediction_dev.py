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
import json
import threading

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
DEV_DATA_LIST = ['../dev.chunk.01']


import glob
import pandas as pd

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np

import time
from contextlib import contextmanager
from functools import partial
@contextmanager
def timer(title):
    print("###### Start {}".format(title))
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# with timer("Load 5epoch model"):
#     model_5 = Doc2Vec.load("../S2B-2018-Kakao-Shopping-Classification/seohyeondeok/word_embedding/tmp/doc2vec_init.5.model")
# def pid2vec(model, x):
#     return model[x.decode()]
# pid2vec = partial(pid2vec, model_5)


# import pickle as pkl
# f=open("tmp/price2vec_200_10_20.pkl", "rb")
# price2vec_model = pkl.load(f)
# f.close()

# def get_price_vec(price_str):
#     try:
#         return price2vec_model[price_str]
#     except:
#         return [0.01]*200
    

class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0

    def get_sample_generator(self, ds, batch_size, raise_stop_event=False):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right] for t in ['embd_word', 'img_feat']]
            
            # X = [ds[t][left:right, :] for t in ['img_feat']]
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

    def write_prediction_result(self, data, pred_y, pred_y_top_n, confid_top_n, meta, out_path, readable):
        pid_order = []
        for data_path in DEV_DATA_LIST:
            h = h5py.File(data_path, 'r')['dev']
            pid_order.extend(h['pid'][::])

            
            
        y2l = {i: s for s, i in six.iteritems(meta['y_vocab'])}
        y2l = list(map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0])))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}

#         with open(out_path, 'w') as fout:
#             for pid, y, y_top_n, p_top_n in zip(data['pid'], pred_y, pred_y_top_n, confid_top_n):
#                 if six.PY3:
#                     pid = pid.decode('utf-8')
#                 aaa = "{}".format(pid)
                
#                 for i, j in zip(y_top_n, p_top_n):
#                     ll = y2l[i] # Lower case of LL
#                     tkns = list(map(int, ll.split('>')))
#                     aaa += "\t{}\t{}\t{}\t{}\t{}".format(j, tkns[0], tkns[1], tkns[2], tkns[3])
                
#                 fout.write(aaa)
#                 fout.write('\n')
#########################################
        pid2label = {}
        for pid, y, y_top_n, p_top_n in zip(data['pid'], pred_y, pred_y_top_n, confid_top_n):
            if six.PY3:
                pid = pid.decode('utf-8')
            aaa = "{}".format(pid)

            for i, j in zip(y_top_n, p_top_n):
                ll = y2l[i] # Lower case of LL
                tkns = list(map(int, ll.split('>')))
                aaa += "\t{}\t{}\t{}\t{}\t{}".format(j, tkns[0], tkns[1], tkns[2], tkns[3])
            pid2label[pid] = aaa
                    
#                 # label = y2l[y]
#                 # tkns = list(map(int, label.split('>')))
#                 # b, m, s, d = tkns
#                 # assert b in inv_cate1['b']
#                 # assert m in inv_cate1['m']
#                 # assert s in inv_cate1['s']
#                 # assert d in inv_cate1['d']
#                 # tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
#                 # if readable:
#                 #     b = inv_cate1['b'][b]
#                 #     m = inv_cate1['m'][m]
#                 #     s = inv_cate1['s'][s]
#                 #     d = inv_cate1['d'][d]
#                 # rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
#                 # ans = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
#                 # fout.write(ans)
#                 # fout.write('\n')
                
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        kkkk = 0 
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
#             for pid in pid2label:
                ans = pid2label.get(pid, no_answer.format(pid=pid))
                # ans = pid2label[pid]
                
                if ans == no_answer.format(pid=pid):
                    kkkk+=1
                fout.write(ans)
                fout.write('\n')
            print("@#$@!#$!@#$@!#$@!#$@#!$ ", kkkk)

    def predict(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        meta_path = os.path.join(data_root, 'meta')
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
        self.write_prediction_result(test, pred_y, pred_y_top_n, confid_top_n, meta, out_path, readable=readable)

    def train(self, data_root, out_dir):
        data_path = os.path.join(data_root, 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path, 'rb').read())
        self.weight_fname = os.path.join(out_dir, 'weights')
        self.model_fname = os.path.join(out_dir, 'model')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.num_classes = len(meta['y_vocab'])

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['cate'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['cate'].shape[0])

        checkpoint = ModelCheckpoint(self.weight_fname, monitor='val_loss',
                                     save_best_only=True, mode='min', period=1)

        textonly = TextOnly()
        model = textonly.get_model(self.num_classes)

        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_sample_generator(train,
                                              batch_size=opt.batch_size)
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_sample_generator(dev,
                                            batch_size=opt.batch_size)
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=[checkpoint])

        model.load_weights(self.weight_fname) # loads from checkout point if exists
        open(self.model_fname + '.json', 'w').write(model.to_json())
        model.save(self.model_fname + '.h5')


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
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict})