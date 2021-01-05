"""
Date: 9/08/2018

Author: Xingjun Ma
Project: elastic_adv_defense
"""
from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, AvgPool2D
from keras.models import Model

def CNN4(input_shape, classes, include_top=True):
    """
    4-layer cnn
    :return:
    """
    img_input = Input(shape=input_shape)

    # LeNet-5 alike 4 layer cnn (removed the second last defense layer)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", name='conv1')(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='pool2')(x)

    # x = Flatten(name='flatten')(x)
    #
    # x = Dense(128, kernel_initializer="he_normal", name='features')(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    if include_top:
        x = Dense(classes, name='logits')(x)
        x = Activation("softmax")(x)

    model = Model(img_input, x)
    return model


def CNN8(input_shape, classes, include_top=True):
    """
    8-layer cnn
    :param input_shape: 
    :param classes: 
    :param include_top: 
    :return: 
    """
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='block1_pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AvgPool2D((2, 2), strides=(2, 2), name='block2_pool1')(x)

    # Block 3
    x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AvgPool2D((2, 2), strides=(2, 2), padding='valid', name='block3_pool1')(x)

    # x = Flatten(name='flatten')(x)

    # x = Dense(256, kernel_initializer="he_normal", name='features')(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    if include_top:
        x = Dense(classes, name='logits')(x)
        x = Activation("softmax")(x)

    model = Model(img_input, x)
    return model

