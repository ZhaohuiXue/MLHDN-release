# tool script
"""
@author: mengxue.zhang
"""

import numpy as np
import cv2
import math
import scipy.io as sio

rd_obj = np.random.RandomState(2020)


def get_data(type='train', mat_path='./input/', shapes=[90, 90]):
    if type == 'train':
        return get_all_data(mat_path, shapes=shapes)
    elif type == 'valid' or type == 'test':
        return get_all_data(mat_path, shapes=shapes, random_flag=False)
    elif type == 'test_gt':
        return get_test_gt(mat_path)
    else:
        print('Invalid type parameter calling get_data!')

def get_batch(type='train', mat_path='./input/', bs=100, shapes=[90, 90]):
   if type=='train':
      return get_batch_data(mat_path, bs=bs, shapes=shapes)
   elif type =='valid' or type =='test':
      return get_batch_data(mat_path, bs=bs, shapes=shapes, random_flag=False)
   else:
      print('Invalid type parameter calling get_batch!')

def get_step(mat_path, bs):
    image_paths, aspects, labels = read_mat2_list(mat_path)
    new_labels = []

    [new_labels.extend(label.tolist()) for label in labels]
    new_labels = np.array(new_labels)
    step = math.ceil(new_labels.shape[0] / bs)
    return step

def get_test_gt(mat_path = ''):
   image_paths, aspects, labels = read_mat2_list(mat_path)
   new_labels = []

   [new_labels.extend(label.tolist()) for label in labels]
   new_labels = np.array(new_labels).squeeze(axis=-1)
   return new_labels

def read_mat2_list(mat_path):
    dict_obj = sio.loadmat(mat_path)
    if dict_obj['images'].shape[0] == 1:
        return dict_obj['images'][0], dict_obj['aspect'][0], dict_obj['labels'][0]
    else:
        return dict_obj['images'], dict_obj['aspect'], dict_obj['labels']

def unwrap(image_paths, aspects, labels):
    new_image_paths = []
    new_aspects = []
    new_labels = []

    [new_image_paths.extend(class_path.tolist()) for class_path in image_paths]
    [new_aspects.extend(aspect.tolist()) for aspect in aspects]
    [new_labels.extend(label.tolist()) for label in labels]
    return np.array(new_image_paths), np.array(new_aspects), np.array(new_labels)

def batch_imread_image(batch_image_paths, aspects=None, sh=[88, 88]):
    flag = True
    for i in range(0, batch_image_paths.shape[0]):
        image_paths = batch_image_paths[i]
        multi_view = np.zeros(shape=[sh[0], sh[1], image_paths.size])
        for j in range(0, image_paths.size):
            item = image_paths[j]
            item = item.rstrip()
            if item.find('\\') != -1:
                item = item.replace('\\', '/')
            try:
                image = cv2.imread(item)
                image = cv2.resize(image, (sh[0], sh[1]), interpolation=cv2.INTER_LINEAR)
                rot_mat = cv2.getRotationMatrix2D((sh[0]//2, sh[1]//2), aspects[i, j], scale=1.0)
                image = cv2.warpAffine(image, rot_mat, (sh[0], sh[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                image = np.power(image[:, :, 0], 1)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                multi_view[:,:,j] = image
            except:
                print(image_paths[j]+' image file is broken!')

        multi_view = np.expand_dims(multi_view, axis=0)
        images = multi_view if flag else np.concatenate((images, multi_view), axis=0)
        flag = False

    images = np.array(images, dtype=np.float16)
    return images

def get_all_data(mat_path='', shapes=[88, 88], random_flag=True):
    image_paths, aspects, labels = read_mat2_list(mat_path)
    image_paths, aspects, labels = unwrap(image_paths, aspects, labels)

    if random_flag:
        random_indexes = np.arange(image_paths.shape[0])
        rd_obj.shuffle(random_indexes)
        image_paths = image_paths[random_indexes]
        aspects = aspects[random_indexes]
        labels = labels[random_indexes]

    images = batch_imread_image(image_paths, aspects=aspects, sh=shapes)
    return images, labels

def get_batch_data(mat_path='', bs=100, shapes=[88, 88], random_flag=True):
    image_paths, aspects, labels = read_mat2_list(mat_path)
    image_paths, aspects, labels = unwrap(image_paths, aspects, labels)
    while True:
        if random_flag:
            random_indexes = np.arange(image_paths.shape[0])
            rd_obj.shuffle(random_indexes)
            image_paths = image_paths[random_indexes]
            aspects = aspects[random_indexes]
            labels = labels[random_indexes]

        for be in range(0, labels.shape[0], bs):
            batch_images_path = image_paths[be:min(be + bs, labels.shape[0])]

            batch_aspects = aspects[be:min(be + bs, labels.shape[0])]
            batch_images = batch_imread_image(batch_images_path, aspects=batch_aspects, sh=shapes)
            batch_labels = labels[be:min(be + bs, labels.shape[0])]

            yield (batch_images, batch_labels)
