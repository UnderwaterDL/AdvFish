from __future__ import absolute_import
from __future__ import print_function

import warnings
import os, math
from tqdm import tqdm
import traceback
import numpy as np
import re
import argparse
import keras
import keras.backend as K
from keras_preprocessing.image import img_to_array, load_img, random_shift, random_rotation, flip_axis
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Dense, Flatten
from keras.models import Model
from resnet import ResNet18, ResNet34
from resnet50 import ResNet50
from efficientnet import EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid error: "OSError: image file is truncated"

from datasets import get_Fish4Knowledge, get_QUTFish, get_WildFish, shuffle

from losses import cross_entropy, symmetric_cross_entropy, margin_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# K.set_image_data_format('channels_first')

def train(dataset='Fish4Knowledge', cnn='resnet50', loss_name='ce', batch_size=32, epochs=50, image_size=32):
    """
    Train one checkpoint with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return:
    """
    print('Dataset: %s, CNN: %s, loss: %s, batch: %s, epochs: %s' % (dataset, cnn, loss_name, batch_size, epochs))

    IMAGE_SIZE = image_size
    INPUT_SHAPE = (image_size, image_size, 3)

    # find image folder: images are distributed in class subfolders
    if dataset == 'Fish4Knowledge':
        # image_path = '/home/xingjun/datasets/Fish4Knowledge/fish_image'
        image_path = '/data/cephfs/punim0619/Fish4Knowledge/fish_image'
        images, labels, images_val, labels_val = get_Fish4Knowledge(image_path, train_ratio=0.8)
    elif dataset == 'QUTFish':
        # image_path = '/home/xingjun/datasets/QUT_fish_data'
        image_path = '/data/cephfs/punim0619/QUT_fish_data'
        images, labels, images_val, labels_val = get_QUTFish(image_path, train_ratio=0.8)
    elif dataset == 'WildFish':
        # image_pathes = ['/home/xingjun/datasets/WildFish/WildFish_part1',
        #             '/home/xingjun/datasets/WildFish/WildFish_part2',
        #             '/home/xingjun/datasets/WildFish/WildFish_part3',
        #             '/home/xingjun/datasets/WildFish/WildFish_part4']
        image_pathes = ['/data/cephfs/punim0619/WildFish/WildFish_part1',
                    '/data/cephfs/punim0619/WildFish/WildFish_part2',
                    '/data/cephfs/punim0619/WildFish/WildFish_part3',
                    '/data/cephfs/punim0619/WildFish/WildFish_part4']
        images, labels, images_val, labels_val = get_WildFish(image_pathes, train_ratio=0.8)

    # images, labels, images_val, labels_val = get_imagenet_googlesearch_data(image_path, num_class=NUM_CLASS)
    num_classes = len(np.unique(labels))
    num_images = len(images)
    num_images_val = len(images_val)
    print('Train: classes: %s, images: %s, val images: %s' %
          (num_classes, num_images, num_images_val))

    global current_index
    current_index = 0

    # dynamic loading a batch of data
    def get_batch():
        index = 1

        global current_index

        B = np.zeros(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        L = np.zeros(shape=(batch_size))
        while index < batch_size:
            try:
                img = load_img(images[current_index], target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img = img_to_array(img)
                img /= 255.
                # if cnn == 'ResNet50': # imagenet pretrained
                #     mean = np.array([0.485, 0.456, 0.406])
                #     std = np.array([0.229, 0.224, 0.225])
                #     img = (img - mean)/std
                ## data augmentation
                # random width and height shift
                img = random_shift(img, 0.2, 0.2)
                # random rotation
                img = random_rotation(img, 10)
                # random horizental flip
                flip_horizontal = (np.random.random() < 0.5)
                if flip_horizontal:
                    img = flip_axis(img, axis=1)
                # # random vertical flip
                # flip_vertical = (np.random.random() < 0.5)
                # if flip_vertical:
                #     img = flip_axis(img, axis=0)
                # #cutout
                # eraser = get_random_eraser(v_l=0, v_h=1, pixel_level=False)
                # img = eraser(img)

                B[index] = img
                L[index] = labels[current_index]
                index = index + 1
                current_index = current_index + 1
            except:
                traceback.print_exc()
                # print("Ignore image {}".format(images[current_index]))
                current_index = current_index + 1
        # B = np.rollaxis(B, 3, 1)
        return B, np_utils.to_categorical(L, num_classes)

    global val_current_index
    val_current_index = 0

    # dynamic loading a batch of validation data
    def get_val_batch():
        index = 1
        B = np.zeros(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        L = np.zeros(shape=(batch_size))

        global val_current_index

        while index < batch_size:
            try:
                img = load_img(images_val[val_current_index], target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img = img_to_array(img)
                img /= 255.
                # if cnn in ['ResNet50', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']: # imagenet pretrained
                #     mean = np.array([0.485, 0.456, 0.406])
                #     std = np.array([0.229, 0.224, 0.225])
                #     img = (img - mean)/std
                B[index] = img
                L[index] = labels_val[val_current_index]
                index = index + 1
                val_current_index = val_current_index + 1
            except:
                traceback.print_exc()
                # print("Ignore image {}".format(images[val_current_index]))
                val_current_index = val_current_index + 1
        # B = np.rollaxis(B, 3, 1)
        return B, np_utils.to_categorical(L, num_classes)


    # load checkpoint
    if cnn == 'ResNet18':
        base_model = ResNet18(input_shape=INPUT_SHAPE, classes=num_classes, include_top=False)
    elif cnn == 'ResNet34':
        base_model = ResNet34(input_shape=INPUT_SHAPE, classes=num_classes, include_top=False)
    elif cnn == 'ResNet50':
        # base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=INPUT_SHAPE)
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    elif cnn == 'EfficientNetB1':
        base_model = EfficientNetB1(input_shape=INPUT_SHAPE, classes=num_classes, include_top=False,
                                    backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif cnn == 'EfficientNetB2':
        base_model = EfficientNetB2(input_shape=INPUT_SHAPE, classes=num_classes, include_top=False,
                                    backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif cnn == 'EfficientNetB3':
        base_model = EfficientNetB3(input_shape=INPUT_SHAPE, classes=num_classes, include_top=False,
                                    backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif cnn == 'EfficientNetB4':
        base_model = EfficientNetB4(input_shape=INPUT_SHAPE, classes=num_classes, include_top=False,
                                    backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    else:
        warnings.warn("Error: unrecognized model!")
        return

    x = base_model.output
    features = Flatten()(x)
    output = Dense(num_classes, activation='softmax')(features)
    model = Model(input=base_model.input, output=output, name=cnn)
    # model.summary()

    if loss_name == 'ce':
        loss = cross_entropy
    elif loss_name == 'sce':
        loss = symmetric_cross_entropy(alpha=1, beta=1.0/num_classes)
    elif loss_name == 'margin':
        loss = margin_loss
    else:
        loss = cross_entropy

    base_lr = 1e-2
    sgd = SGD(lr=base_lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss=loss,
        optimizer=sgd,
        metrics=['accuracy']
    )

    # always save your weights after training or during training
    # create folder if not exist
    if not os.path.exists('models/'):
        os.makedirs('models/')
    log_path = 'log/%s' % dataset
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ## loop the weight folder then load the lastest weight file continue training
    model_prefix = '%s_%s_%s_' % (dataset, cnn, loss_name)
    w_files = os.listdir('models/')
    existing_ep = 0
    # for fl in w_files:
    #     if model_prefix in fl:
    #         ep = re.search(model_prefix+"(.+?).h5", fl).group(1)
    #         if int(ep) > existing_ep:
    #             existing_ep = int(ep)
    #
    # if existing_ep > 0:
    #     weight_file = 'models/' + model_prefix + str(existing_ep) + ".h5"
    #     print("load previous model weights from: ", weight_file)
    #     model.load_weights(weight_file)
    #
    #     log = np.load(os.path.join(log_path, 'train_log_%s_%s.npy' % (cnn, loss_name)))
    #
    #     train_loss_log = log[0, :existing_ep+1].tolist()
    #     train_acc_log = log[1, :existing_ep+1].tolist()
    #     val_loss_log = log[2, :existing_ep+1].tolist()
    #     val_acc_log = log[3, :existing_ep+1].tolist()
    # else:

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    # dynamic training
    for ep in range(epochs-existing_ep):
        ## cosine learning rate annealing
        eta_min = 0.
        eta_max = base_lr
        lr = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * ep / epochs)) / 2
        K.set_value(model.optimizer.lr, lr)
        # # step-wise learning rate annealing
        # if ep in [int(epochs*0.5), int(epochs*0.75)]:
        #     lr = K.get_value(model.optimizer.lr)
        #     K.set_value(model.optimizer.lr, lr*.1)
        #     print("lr decayed to {}".format(lr*.1))

        current_index = 0
        n_step = int(num_images/batch_size)
        pbar = tqdm(range(n_step))
        for stp in pbar:
            b, l = get_batch()
            train_loss, train_acc = model.train_on_batch(b, l)
            pbar.set_postfix(acc='%.4f' % train_acc, loss='%.4f' % train_loss)

        ## test acc and loss at each epoch
        val_current_index = 0
        y_pred = []
        y_true = []
        while val_current_index + batch_size < num_images_val:
            b, l = get_val_batch()
            pred = model.predict(b)
            y_pred.extend(pred.tolist())
            y_true.extend(l.tolist())

        y_pred = np.clip(np.array(y_pred), 1e-7, 1.)
        correct_pred = (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        val_acc = np.mean(correct_pred)
        val_loss = -np.sum(np.mean(y_true * np.log(y_pred), axis=1))/val_current_index

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)
        log = np.stack((np.array(train_loss_log),
                        np.array(train_acc_log),
                        np.array(val_loss_log),
                        np.array(val_acc_log)))

        # save training log
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        np.save(os.path.join(log_path, 'train_log_%s_%s.npy' % (cnn, loss_name)), log)

        pbar.set_postfix(acc='%.4f' % train_acc, loss='%.4f' % train_loss,
                         val_acc='%.4f' % val_acc, val_loss='%.4f' % val_loss)
        print("Epoch %s - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f"
              % (ep, train_loss, train_acc, val_loss, val_acc))
        images, labels = shuffle(images, labels)
        if ((ep+existing_ep+1)%5 == 0) or (ep == (epochs - existing_ep - 1)):
            model_file = 'models/%s_%s_%s_%s.h5' % (dataset, cnn, loss_name, ep+existing_ep)
            model.save_weights(model_file)

    # # evaluate the checkpoint
    # val_current_index = 0
    # val_loss = 0.0
    # val_acc = 0.0
    # while val_current_index + batch_size < num_images_val:
    #     b, l = get_val_batch()
    #
    #     score = model.test_on_batch(b, l)
    #     print('batch %s/%s val_loss: %.2f' % (current_index // batch_size,
    #                                           nice_n // batch_size, score[0]))
    #     print('batch %s/%s val_acc: %.2f' % (current_index // batch_size,
    #                                          nice_n // batch_size, score[1]))
    #
    #     val_loss += score[0]
    #     val_acc += score[1]
    #
    # val_loss = val_loss / (nice_n // batch_size)
    # val_acc = val_acc / (nice_n // batch_size)
    #
    # print('Mean val_loss: %.2f' % val_loss)
    # print('Mean val_acc: %.2f' % val_acc)

def main(args):
    train(args.dataset, args.model, args.loss_name, args.batch_size, args.epochs, args.image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'Fish4Knowledge', QUTFish or 'WildFish'",
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--model',
        help="The CNN used for training;: 'ResNet18', 'ResNet34', 'ResNet50',"
             "'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4'",
        required=True, type=str
    )
    parser.add_argument(
        '-l', '--loss_name',
        help="The loss function used for training: ce, sce, margin",
        required=True, type=str)
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-s', '--image_size',
        help="Size of the image.",
        required=False, type=int
    )
    parser.add_argument(
        '-c', '--num_class',
        help="Number of classes.",
        required=False, type=int
    )
    parser.set_defaults(epochs=50)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(image_size=64)

    args = parser.parse_args()
    main(args)

    # args = parser.parse_args(['-d', 'Fish4Knowledge', '-m', 'ResNet50', '-l', 'ce',
    #                           '-e', '50', '-b', '32', '-s', '224'])
    # main(args)

    # args = parser.parse_args(['-d', 'QUTFish', '-m', 'EfficientNetB3', '-l', 'margin',
    #                           '-e', '50', '-b', '32', '-s', '224'])
    # main(args)

    # args = parser.parse_args(['-d', 'WildFish', '-m', 'ResNet18', '-l', 'ce',
    #                           '-e', '50', '-b', '32', '-s', '224'])
    # main(args)
