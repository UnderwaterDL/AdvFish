# -*- coding: utf-8 -*-
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class AMSoftmax(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(AMSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.kernel = K.l2_normalize(self.kernel, 0)

    def call(self, inputs, **kwargs):
        inputs = K.l2_normalize(inputs, -1)  # input_l2norm
        output = K.dot(inputs, self.kernel)   # cos = input_l2norm * W_l2norm
        return output

    def compute_output_shape(self, input_shape):
        outputshape = list(input_shape)
        outputshape[-1] = self.units
        return tuple(outputshape)


def amsoftmax_loss(y_true, y_pred, scale=30.0,
    margin=0.35):

    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale

    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
