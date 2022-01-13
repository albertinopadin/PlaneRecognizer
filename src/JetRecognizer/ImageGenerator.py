import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, img_to_array
from PIL import Image
from ImagePreprocessing.ImagePreprocessing import JPG_EXT, convert_labels_to_one_hot_vectors


NORMALIZATION_LAYER = tf.keras.layers.Rescaling(1./255)


def create_img_tensor(filename):
    img = Image.open(filename)
    return img_to_array(img)


def load_image_tensors_with_labels_in_dir_generator(directory, img_type=JPG_EXT, shuffle=True):
    img_filenames_w_labels = []
    for folder, _, fnames in os.walk(directory):
        for fname in fnames:
            basepth = os.path.basename(folder)
            if directory != basepth:
                label = basepth[:-1]
                fn_w_label = (f'{folder}/{fname}', label)
                img_filenames_w_labels.append(fn_w_label)

    if shuffle:
        random.shuffle(img_filenames_w_labels)

    def img_tensor_label_gen():
        for fn, lbl in img_filenames_w_labels:
            if fn.endswith(img_type):
                img_tensor = create_img_tensor(fn)
                yield img_tensor, lbl

    return img_tensor_label_gen(), len(img_filenames_w_labels)


def load_random_cropped_normalized_image_tensors_w_labels_in_dir_generator(directory, crop_size, img_type=JPG_EXT):
    load_img_tensor_generator, total_len = load_image_tensors_with_labels_in_dir_generator(directory, img_type)

    def random_cropped_normalized_gen():
        for img_tensor, label in load_img_tensor_generator:
            normalized = NORMALIZATION_LAYER(img_tensor)
            x, y, _ = normalized.shape
            crop_width, crop_height, _ = crop_size
            if x < crop_width or y < crop_height:
                normalized_cropped = tf.image.resize_with_crop_or_pad(normalized, crop_width, crop_height)
            else:
                normalized_cropped = tf.image.random_crop(normalized, size=crop_size)

            yield normalized_cropped, label

    return random_cropped_normalized_gen(), total_len


@tf.function
def random_flips(batch_img_tensors):
    with tf.device('/GPU:0'):
        batch_img_tensors = tf.image.random_flip_left_right(batch_img_tensors, seed=random.randint(0, 100))
        batch_img_tensors = tf.image.random_flip_up_down(batch_img_tensors, seed=random.randint(0, 100))
    return batch_img_tensors


class ImageGenerator(Sequence):
    LEN_OFFSET = 5  # The generator runs empty early for some reason - using this as a hack while I figure out why

    def __init__(self, img_dir, crop_size=(862,862,3), batch_size=2, label_encoder=None, one_hot_labels=False):
        if one_hot_labels:
            if label_encoder is None:
                raise Exception('Must set label encoder to one-hot encode labels.')

        self.img_dir = img_dir
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.label_encoder = label_encoder
        self.one_hot_labels = one_hot_labels
        self._generator, self.total_len = \
            load_random_cropped_normalized_image_tensors_w_labels_in_dir_generator(img_dir, crop_size=crop_size)

    def __len__(self):
        return int(np.floor(self.total_len/self.batch_size)) - ImageGenerator.LEN_OFFSET

    def __getitem__(self, index):
        batch_img_tensors = []
        batch_labels = []
        for _ in range(self.batch_size):
            try:
                img_tensor, label = next(self._generator)
                batch_img_tensors.append(img_tensor)
                batch_labels.append(label)
            except StopIteration:
                break

        batch_img_tensors = random_flips(batch_img_tensors)

        if self.one_hot_labels:
            batch_labels, _ = convert_labels_to_one_hot_vectors(batch_labels, encoder=self.label_encoder)

        return batch_img_tensors, batch_labels

    def on_epoch_end(self):
        self._generator, self.total_len = \
            load_random_cropped_normalized_image_tensors_w_labels_in_dir_generator(self.img_dir, crop_size=self.crop_size)