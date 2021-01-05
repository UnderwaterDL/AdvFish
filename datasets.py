"""
Get the train and val subset at the same time for a fish dataset.
Each of the following get data function returns two data.Dataset instances: train, val.
- The reason to return train and val at the same time
is because the files splitting (into train/val) should be done only once
to avoid overlapping files in train and val.

"""
import os
import numpy as np
from random import sample
import random
import warnings

from PIL import Image
import piexif

# fix random seed to make sure train/test samples are consistent when load again
random.seed(1234)

def get_Fish4Knowledge(image_path, train_ratio=0.8):
    """
    get train and test dataset of Fish4Knowledge:
    http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/
    step1: download the dataset
    step2: set the root to fish_image/

    :param image_path: the Fish4Knowledge/fish_image/
    :param the percentage used for training
    :return:
    """
    # if the images has been scanned before then just load
    train_images_file = 'data/Fish4Knowledge_train_images.npy'
    train_labels_file = 'data/Fish4Knowledge_train_labels.npy'
    test_images_file = 'data/Fish4Knowledge_test_images.npy'
    test_labels_file = 'data/Fish4Knowledge_test_labels.npy'
    if os.path.exists(train_images_file):
        print('Found pre-generated train/test lists!')
        images_train = np.load(train_images_file)
        labels_train = np.load(train_labels_file)
        images_val = np.load(test_images_file)
        labels_val = np.load(test_labels_file)
        images_train, labels_train = shuffle(images_train, labels_train)
        return images_train, labels_train, images_val, labels_val

    # scan the image folder to get the train and test image/label list
    images = []
    labels = []
    label_id = 0
    for sub in os.listdir(image_path):
        sub_dir = os.path.join(image_path, sub)
        image_list = os.listdir(sub_dir)
        for image in image_list:
            images.append(os.path.join(sub_dir, image))
            labels.append(label_id)

        print('Dataset [Fish4Knowledge]: #class=%s, #sample=%s' % (label_id, len(image_list)))

        label_id += 1

    images_train, labels_train, images_val, labels_val = train_val_split(images, labels, train_ratio)

    # save the indexes to files
    np.save(train_images_file, np.asarray(images_train))
    np.save(train_labels_file, np.asarray(labels_train))
    np.save(test_images_file, np.asarray(images_val))
    np.save(test_labels_file, np.asarray(labels_val))

    # random shuffle
    images_train, labels_train = shuffle(images_train, labels_train)

    return images_train, labels_train, images_val, labels_val


def get_QUTFish(image_path, train_ratio=0.8):
    """
    get train and test dataset of QUTFish:
    https://wiki.qut.edu.au/display/cyphy/Fish+Dataset
    step1: download the dataset
    step2: set the root to QUT_fish_data/

    :param image_path: the QUT_fish_data/
    :param the percentage used for training
    :return:
    """
    # if the images has been scanned before then just load
    train_images_file = 'data/QUTFish_train_images.npy'
    train_labels_file = 'data/QUTFish_train_labels.npy'
    test_images_file = 'data/QUTFish_test_images.npy'
    test_labels_file = 'data/QUTFish_test_labels.npy'
    if os.path.exists(train_images_file):
        print('Found pre-generated train/test lists!')
        images_train = np.load(train_images_file)
        labels_train = np.load(train_labels_file)
        images_val = np.load(test_images_file)
        labels_val = np.load(test_labels_file)
        images_train, labels_train = shuffle(images_train, labels_train)
        return images_train, labels_train, images_val, labels_val


    # scan the image folder to get the train and test image/label list
    images = []
    labels = []
    label_id = 0
    # read label and image file list from final_all_index.txt
    # line format: 1=A73EGS~P=controlled=A73EGS~P_7=7s
    images_tmp = []
    current_class = None
    with open(os.path.join(image_path, "final_all_index.txt")) as f:
        for line in f:
            names = line.split('=')
            if names[2] != 'insitu':
                continue
            if not os.path.exists(os.path.join(image_path, 'images/raw_images/' + names[3] + '.jpg')):
                continue
            # print(names)

            if current_class is None:
                current_class = int(names[0])
                images_tmp.append(os.path.join(image_path, 'images/raw_images/' + names[3] + '.jpg'))
            else:
                if current_class == int(names[0]):
                    images_tmp.append(os.path.join(image_path, 'images/raw_images/' + names[3] + '.jpg'))
                else:
                    if len(images_tmp) > 10: # only save class has >10 images
                        # append this class to dataset
                        labels_tmp = np.ones(len(images_tmp))*label_id
                        images.extend(images_tmp)
                        labels.extend(labels_tmp.astype(np.int8).tolist())
                        label_id += 1
                        print('Dataset [QUTFish]: #class=%s, #sample=%s' % (label_id, len(images_tmp)))

                    # move on to next class
                    current_class = int(names[0])
                    images_tmp = []
                    images_tmp.append(os.path.join(image_path, 'images/raw_images/' + names[3] + '.jpg'))

    print('QUT: #classes: ', label_id, ', #images: ', len(images))

    images_train, labels_train, images_val, labels_val = train_val_split(images, labels, train_ratio)

    # save the indexes to files
    np.save(train_images_file, np.asarray(images_train))
    np.save(train_labels_file, np.asarray(labels_train))
    np.save(test_images_file, np.asarray(images_val))
    np.save(test_labels_file, np.asarray(labels_val))

    # random shuffle
    images_train, labels_train = shuffle(images_train, labels_train)

    return images_train, labels_train, images_val, labels_val



def get_WildFish(image_pathes, train_ratio=0.8):
    """
    get train and test dataset of WildFish:
    https://github.com/PeiqinZhuang/WildFish
    step1: download the dataset into 4 folders: WildFish_part1/2/3/4
    step2: set the root folders (a list) to WildFish_part1/2/3/4

    :param image_pathes:
    :param train_ratio:
    :return:
    """
    warnings.warn('For WilfFish dataset, please run dataset.py -> clean_wildfish()'
                  ' to clean the dataset first!!')
    # if the images has been scanned before then just load
    train_images_file = 'data/WildFish_train_images.npy'
    train_labels_file = 'data/WildFish_train_labels.npy'
    test_images_file = 'data/WildFish_test_images.npy'
    test_labels_file = 'data/WildFish_test_labels.npy'
    if os.path.exists(train_images_file):
        print('Found pre-generated train/test lists!')
        images_train = np.load(train_images_file)
        labels_train = np.load(train_labels_file)
        images_val = np.load(test_images_file)
        labels_val = np.load(test_labels_file)
        images_train, labels_train = shuffle(images_train, labels_train)
        return images_train, labels_train, images_val, labels_val

    # scan the image folder to get the train and test image/label list
    # get all the fish names first as different fishes are all stored in one folder
    fish_names = set()
    for part in image_pathes:
        files = os.listdir(part)
        for f in files:
            s = f.split('_')
            name = s[0] + '_' + s[1]
            fish_names.add(name)
    fish_names = list(fish_names)
    fish_names.sort()
    # print(fish_names)

    print('#classes: %s' % len(fish_names))
    print(fish_names[:10])

    ## loop again to list the images and labels
    images = []
    labels = []
    for part in image_pathes:
        files = os.listdir(part)
        for f in files:
            s = f.split('_')
            name = s[0] + '_' + s[1]
            cls = fish_names.index(name)

            image_file = os.path.join(part, f)

            # append to the list
            images.append(image_file)
            labels.append(cls)

    print('Dataset [WildFish]: #class=%s, #sample=%s' % (len(fish_names), len(images)))

    images_train, labels_train, images_val, labels_val = train_val_split(images, labels, train_ratio)

    # save the indexes to files
    np.save(train_images_file, np.asarray(images_train))
    np.save(train_labels_file, np.asarray(labels_train))
    np.save(test_images_file, np.asarray(images_val))
    np.save(test_labels_file, np.asarray(labels_val))

    # random shuffle
    images_train, labels_train = shuffle(images_train, labels_train)

    return images_train, labels_train, images_val, labels_val

def train_val_split(images, labels, train_ratio=0.8):
    """
    Stratified split (class-wise) file_names and targets into train/val set, according to the train_ratio
    :param images:
    :param labels: should start from 0
    :param train_ratio: default 0.8
    :return: images_train, labels_train, images_val, labels_val
    """
    images_train, labels_train, images_val, labels_val = [], [], [], []

    cls_list = [[] for i in range(np.max(labels) + 1)]
    for x, y in zip(images, labels):  # seperate into classes
        cls_list[y].append(x)

    for cls, file_list in enumerate(cls_list):
        n_sample = len(file_list)
        n_train = int(n_sample * train_ratio)
        # n_val = n_sample - n_train

        data_train = sample(file_list, n_train)
        for i in data_train:
            images_train.append(str(i))
            labels_train.append(cls)

        for j in file_list:
            if j not in data_train:
                images_val.append(str(j))
                labels_val.append(cls)

    return images_train, labels_train, images_val, labels_val


def shuffle(images, labels):
    """
    shuffle the images and its labels
    :param images:
    :param labels:
    :return:
    """
    # shuffle
    perm = list(range(len(images)))
    np.random.shuffle(perm)
    images = [images[idx] for idx in perm]
    labels = [labels[idx] for idx in perm]
    return images, labels

def clean_wildfish(path):
    # scan the image folder to clean each .jpg, jpeg file
    count = 0
    for part in path:
        files = os.listdir(part)
        for f in files:
            image_file = os.path.join(part, f)
            if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG'):
                if f.endswith('.jpeg') or f.endswith('.JPEG'):
                    piexif.remove(image_file)
                # following is the alternative approach
                try:
                    image = Image.open(image_file)
                    if image.mode != 'RGB':
                        image = image.convert("RGB")
                    image.save(image_file)
                except:
                    # delete this file
                    print('corrupted file deleted: ', image_file)
                    os.remove(image_file)
                    continue
                count += 1
                if count%1000 == 0:
                    print(count, 'images processed!')

def replace_path_in_train_test(old_path, new_path, train_test_file):
    """
    change the root path in train_test_file from old -> new
    :param old_path:
    :param new_path:
    :param file:
    :return:
    """
    img_files = np.load(train_test_file)
    img_files = [sub.replace(old_path, new_path) for sub in img_files]
    np.save(train_test_file, np.asarray(img_files))
    print('Replacing path done for: ', train_test_file)


if __name__ == "__main__":
    # path = ['/home/xingjun/datasets/WildFish/WildFish_part1',
    #         '/home/xingjun/datasets/WildFish/WildFish_part2',
    #         '/home/xingjun/datasets/WildFish/WildFish_part3',
    #         '/home/xingjun/datasets/WildFish/WildFish_part4']
    # path = ['/data/cephfs/punim0619/WildFish/WildFish_part1',
    #         '/data/cephfs/punim0619/WildFish/WildFish_part2',
    #         '/data/cephfs/punim0619/WildFish/WildFish_part3',
    #         '/data/cephfs/punim0619/WildFish/WildFish_part4']
    # clean_wildfish(path)

    old_path = '/home/xingjun/datasets'
    new_path = '/data/cephfs/punim0619'
    file_list = ['data/Fish4Knowledge_train_images_2.npy',
                 'data/Fish4Knowledge_test_images_2.npy',
                 'data/QUTFish_train_images_2.npy',
                 'data/QUTFish_test_images_2.npy',
                 'data/WildFish_train_images_2.npy',
                 'data/WildFish_test_images_2.npy']
    # file_list = [
    #              'data/QUTFish_train_images_2.npy']
    for train_test_file in file_list:
        replace_path_in_train_test(old_path, new_path, train_test_file)

    # img_files = np.load('data/QUTFish_train_images_2.npy')
    # for x in img_files:
    #     print(x)
