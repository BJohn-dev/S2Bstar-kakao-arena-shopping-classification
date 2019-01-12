# 
# inference.py
# ==============================================================================
import os
import sys
import json


import h5py
import tqdm
import numpy as np
import six

from keras.models import load_model
from six.moves import zip, cPickle

from network import MainNet, top1_acc, customLoss

from utils_class import ClassifierBone, ThreadsafeIter, opt, cate1
from utils_post import dict_max, post_process_first_stage, ml_changer, \
                       rank_puller, get_new_s_pre, get_new_d_pre, \
                       get_new_s_pre_only_b_pre, get_new_d_pre_only_b_pre


class Infer(ClassifierBone):

    def write_prediction_result(self, data, pred_y, pred_y_top_n, confid_top_n, meta, out_path, dataset_name):
        
        DEV_DATA_LIST = None
        if dataset_name == 'dev':
            DEV_DATA_LIST = ['../dev.chunk.01']
        elif dataset_name == 'test':
            DEV_DATA_LIST = ['../test.chunk.01', '../test.chunk.02']
            
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
            
            #################################################################################
            #################### MLE
            aaa_list = post_process_first_stage(aaa_list)
            aaa_list = ml_changer(aaa_list)
            pre_processed_list = rank_puller(aaa_list, max_rank=5)
            
            #################################################################################
            #################### MLE + MAP
            for index in range(1, len(pre_processed_list)):
                if int(pre_processed_list[index]) == -1:
                    pre_processed_list[index] = int(aaa_list_for_map[index])

            pid2label[pid] = "\t".join(list(map(str, pre_processed_list)))
                                       
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
                ans = pid2label.get(pid, no_answer.format(pid=pid))

                fout.write(ans)
                fout.write('\n')

    def predict(self, dataset_name):
            
        test_root = 'data/{}/'.format(dataset_name)
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
        total_test_samples = test['embd_word'].shape[0]
        
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['embd_word'].shape[0]
                X, _ = chunk
                _pred_y = model.predict(X)
                pred_y.extend([np.argmax(y) for y in _pred_y])
    
                pred_y_top_n.extend([np.argsort(y)[-n:][::-1] for y in _pred_y])
                confid_top_n.extend([y[np.argsort(y)[-n:][::-1]] for y in _pred_y])
                
                pbar.update(X[0].shape[0])
        self.write_prediction_result(test, pred_y, pred_y_top_n, confid_top_n, meta, out_path, dataset_name)


if __name__ == '__main__':
    dataset_name = sys.argv[1]    
    infer = Infer('Inference')
    infer.predict(dataset_name)

