# 
# network.py
# ==============================================================================
import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input, Concatenate
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)

def customLoss(y_true,y_pred):
    return y_true


class MainNet:
    def __init__(self):
        self.logger = get_logger('main_net')

    def get_model(self, num_classes, activation='softmax'):

        with tf.device('/gpu:0'):
            
            ######################################################
            embd_word = Input((opt.embd_size,), name="embd_word")
            output_word = Dense(int(num_classes*2), activation='relu', name="outputs_word")(Dropout(rate=0.4)(embd_word))
            
            ######################################################
            img_feat = Input((opt.img_size,), name="img_feat")
            outputs_feat = Dense(int(opt.img_size), activation='relu', name="outputs_img_feat")(Dropout(rate=0.4)(img_feat))
            
            ######################################################
            added = Concatenate(axis=-1)([output_word, outputs_feat])
            outputs_sum = Dense(int(num_classes), activation='relu', name="outputs_sum")(Dropout(rate=0.6)(added))
            
            ######################################################
            outputs = Dense(num_classes, activation=activation, name="outputs_dense")(Dropout(rate=0.6)(outputs_sum))
            
            ######################################################
            model = Model(inputs=[embd_word, img_feat], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr_train)
            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class EmbdNet:
    def __init__(self):
        self.logger = get_logger('embd_net')

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
            embd_out = Dropout(rate=0.5)(uni_embd)
            
            ######################################################
            word = Activation('relu', name='relu1')(embd_out)
            output_word = Dense(int(num_classes), activation='relu', name="outputs_word")(Dropout(rate=0.4)(word))
            
            ######################################################
            outputs_sum = Dense(int(num_classes*3), activation='relu', name="outputs_sum")(Dropout(rate=0.6)(output_word))
            
            ######################################################
            outputs = Dense(num_classes, activation=activation, name="outputs_dense")(Dropout(rate=0.6)(outputs_sum))
            
            ######################################################
            model = Model(inputs=[t_uni, w_uni], outputs=[outputs, embd_out])
            optm = keras.optimizers.Nadam(opt.lr_embd)
            model.compile(loss=['categorical_crossentropy', customLoss],
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model