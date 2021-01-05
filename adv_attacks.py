"""
Adapted from:
https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py

Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class LinfPGDAttack:
    def __init__(self, model, epsilon, eps_iter, nb_iter, kappa=0, random_start=False,
                 loss_func='xent', clip_min=0.0, clip_max=1.0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.kappa = kappa
        self.rand = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.x_input = self.model.layers[0].input
        logits = self.model.layers[-2].output
        y_pred = tf.nn.softmax(logits)
        self.y_true = tf.placeholder(tf.float32, shape=y_pred.get_shape().as_list())

        if loss_func == 'xent':
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)
        elif loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.y_true * logits, axis=1)
            wrong_logit = tf.reduce_max((1 - self.y_true) * logits, axis=1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + kappa)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)

        self.grad = tf.gradients(self.loss, self.x_input)[0]

    def perturb(self, sess, x_nat, y, batch_size):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        nb_batch = len(x) // batch_size
        # check if need one more batch
        if nb_batch * batch_size < len(x):
            nb_batch += 1
        
        for i in range(nb_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = np.minimum(end, len(x))
            batch_x = x[start:end]
            batch_y = y[start:end]
            for j in range(self.nb_iter):
                loss, grad = sess.run([self.loss, self.grad],
                                      feed_dict={self.x_input: batch_x,
                                                 self.y_true: batch_y})
                grad = np.nan_to_num(grad)
                batch_x += self.eps_iter * np.sign(grad)
                batch_x = np.clip(batch_x, x_nat[start:end] - self.epsilon, x_nat[start:end] + self.epsilon)
                batch_x = np.clip(batch_x, self.clip_min, self.clip_max)  # ensure valid pixel range

            x[start:end] = batch_x[:]

        return x


"""
Adaptive Fast Gradient Sign Method (AdaFGSM)
"""
class AdaFGSM:
    def __init__(self, model, epsilon, kappa=0, random_start=False,
                 loss_func='xent', clip_min=0.0, clip_max=1.0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.kappa = kappa
        self.rand = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.x_input = self.model.layers[0].input
        logits = self.model.layers[-2].output
        y_pred = tf.nn.softmax(logits)
        self.y_true = tf.placeholder(tf.float32, shape=y_pred.get_shape().as_list())

        if loss_func == 'xent':
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)
        elif loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.y_true * logits, axis=1)
            wrong_logit = tf.reduce_max((1 - self.y_true) * logits, axis=1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + kappa)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)

        self.grad = tf.gradients(self.loss, self.x_input)[0]

    def perturb(self, sess, x_nat, y, batch_size):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        nb_batch = len(x) // batch_size
        # check if need one more batch
        if nb_batch * batch_size < len(x):
            nb_batch += 1

        for i in range(nb_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = np.minimum(end, len(x))
            batch_x = x[start:end]
            batch_y = y[start:end]

            # compute the input gradients
            loss, grad = sess.run([self.loss, self.grad],
                                  feed_dict={self.x_input: batch_x,
                                             self.y_true: batch_y})
            grad = np.nan_to_num(grad)

            # get the maximum gradient magnitude
            max_norm = np.max(np.abs(grad))
            # normalized gradient to [0,1]
            grad = grad/max_norm
            # apply an adaptive perturbation to input, replacing the hard perturbation: epsilon*sign(grad)
            batch_x += self.epsilon * grad

            # clipping to ensure the perturbed pixel values are still within the valid range (eg. [0,1])
            batch_x = np.clip(batch_x, x_nat[start:end] - self.epsilon, x_nat[start:end] + self.epsilon)
            batch_x = np.clip(batch_x, self.clip_min, self.clip_max)  # ensure valid pixel range [0,1]

            x[start:end] = batch_x[:]

        return x
