# 
# train_predict.py
# ==============================================================================
import os
import fire
import tqdm
import six
import h5py
from six.moves import zip, cPickle

import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from utils_class import ClassifierBone, ThreadsafeIter, opt, cate1
from network import top1_acc, customLoss, EmbdNet
    

class Trainer(ClassifierBone):
    pass


class Predictor(ClassifierBone):
    pass


class EmbdTrainer(ClassifierBone):
    
    def get_sample_generator(self, ds, batch_size, raise_stop_event=False):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right] for t in ['uni', 'w_uni']]
            
            Y = [ds['cate'][left:right], 
                 np.zeros((right-left, opt.embd_size))]
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration

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

        checkpoint = ModelCheckpoint(self.weight_fname, 
                                     monitor='val_loss',
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)

        embd_net = EmbdNet()
        model = embd_net.get_model(self.num_classes)

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
                            epochs=opt.num_epochs_embd,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=[checkpoint])
   
    
if __name__ == '__main__':
    trainer = Trainer('Trainer')
    predictor = Predictor('Predictor')
    embd_trainer = EmbdTrainer('EmbdTrainer')
    fire.Fire({'train': trainer.train, 
               'predict': predictor.predict, 
               'embd_train': embd_trainer.train})
