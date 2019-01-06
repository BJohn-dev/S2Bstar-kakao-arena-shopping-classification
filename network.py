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


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):

        with tf.device('/gpu:0'):
            
            ######################################################
            embd_word = Input((opt.embd_size,), name="embd_word")
            output_word = Dense(int(num_classes*2), activation='relu', name="outputs_word")(Dropout(rate=0.4)(embd_word))
            
            ######################################################
            img_feat = Input((2048,), name="img_feat")
            outputs_feat = Dense(int(2048), activation='relu', name="outputs_img_feat")(Dropout(rate=0.4)(img_feat))
            
            ######################################################
            added = Concatenate(axis=-1)([output_word, outputs_feat])
            outputs_sum = Dense(int(num_classes), activation='relu', name="outputs_sum")(Dropout(rate=0.6)(added))
            
            ######################################################
            activation = 'softmax'
            outputs = Dense(num_classes, activation=activation, name="outputs_dense")(Dropout(rate=0.6)(outputs_sum))
            
            ######################################################
            model = Model(inputs=[embd_word, img_feat], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr_train)
            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
