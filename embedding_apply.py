import os
import sys

import h5py
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from six.moves import zip, cPickle

from embedding_network import customLoss, top1_acc


model_root = "embedding_model/train"
model_fname = os.path.join(model_root, 'weights')

embd_model = load_model(model_fname,
                   custom_objects={'top1_acc': top1_acc, 'customLoss': customLoss})

embd_size = 1254
batch_size = 500000
    
data_root = 'data/'

dataset_name = sys.argv[1]
div_list = None
if dataset_name  == 'train':
    div_list = ['train', 'dev']
elif dataset_name in ['dev', 'test']:
    div_list = ['dev']

    
for div in div_list:
    data_set_root = os.path.join(data_root, dataset_name)
    data_set_path = os.path.join(data_set_root, 'data.h5py')

    data_set = h5py.File(data_set_path, 'a')

    size = data_set[div]['img_feat'].shape[0]
    num_iter = int(size/batch_size)+1

    data_set[div].create_dataset('embd_word', (size, embd_size), chunks=True, dtype=np.float32)
    for i in range(num_iter):

        s_index = i*batch_size
        f_index = (i+1)*batch_size

        uni = data_set[div]['uni'][s_index:f_index]
        w_uni = data_set[div]['w_uni'][s_index:f_index]

        rets = embd_model.predict([uni, w_uni])


        data_set[div]['embd_word'][s_index:f_index, :] = rets[1]

        print(i, "/", num_iter)
        
    data_set.close()
