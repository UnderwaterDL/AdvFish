import numpy as np
from keras import backend as K
import tensorflow as tf

def cross_entropy(y_true, y_pred):
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
    return -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)

def mix_ce_loss(y_true, y_pred):
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())

    full = tf.shape(y_pred)[0]
    half = tf.to_int32(full/2)

    # the following works beter
    l2_loss = tf.reduce_sum(y_pred[half:] * K.log(y_pred[half:]/(1-y_pred[:half])), axis=1)

    ce_loss = -tf.reduce_sum(y_true[:half] * tf.log(y_pred[:half]), axis=1)

    # the loss should have the full batch size
    # use tf.concat to achieve that
    loss_comb = tf.concat([ce_loss, 0.01*l2_loss], axis=0)
    final_loss = tf.reduce_mean(loss_comb, axis=-1)
    return final_loss

def symmetric_cross_entropy(alpha, beta):
    """
    Symmetric Cross Entropy for Robust Learning with Noisy Labels:
    https://arxiv.org/abs/1908.06112
    https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
    :param alpha:
    :param beta:
    :return:
    """
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis=-1)) +\
               beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis=-1))
    return loss


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))
