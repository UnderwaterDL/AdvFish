import numpy as np
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

def zero_loss(y_true, y_pred):
    return K.mean(y_pred, axis=0)

class CenterLossLayer(Layer):
    def __init__(self, alpha=0.5, num_classes=10, num_features=256, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_classes, self.num_features),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is Nxn_features, x[1] is Nxn_classes onehot, self.centers is n_classesxn_features
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
