import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from os.path import isfile, join
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import random
from Common.DL_FilePaths import PROJECT_ROOT


JPG_EXT = '.jpg'
PREPROCESSED_FOLDER_NAME = 'preprocessed'
NUM_CPUS = cpu_count()
NUM_THREADS = 4


def get_file_paths_in_dir(directory, exclude_subfolders=True):
    if exclude_subfolders:
        return [f'{directory}/{fn}' for fn in os.listdir(directory) if isfile(join(directory, fn))]
    else:
        return [f'{directory}/{fn}' for fn in os.listdir(directory)]


def create_preprocessed_image_dir(dir_name):
    preprocessed_dir_name = f'{dir_name}/{PREPROCESSED_FOLDER_NAME}'
    if not os.path.exists(preprocessed_dir_name):
        os.mkdir(preprocessed_dir_name)
    return preprocessed_dir_name


def get_largest_image_dimensions_in_dir(directory):
    print(f'Getting largest image dimensions in {directory}')
    paths_in_dir = get_file_paths_in_dir(directory)
    largest_width = 0
    largest_height = 0

    for path in paths_in_dir:
        if path.endswith(JPG_EXT):
            img = Image.open(path)
            width, height = img.size
            if width > largest_width:
                largest_width = width
            if height > largest_height:
                largest_height = height

    return largest_width, largest_height


# TODO: Maybe this is worth parallelizing:
def get_largest_image_dimensions_in_directories(directory_list):
    largest_width = 0
    largest_height = 0
    for directory in directory_list:
        l_width, l_height = get_largest_image_dimensions_in_dir(directory)
        if l_width > largest_width:
            largest_width = l_width
        if l_height > largest_height:
            largest_height = l_height

    return largest_width, largest_height


def pad_pil_image(img, desired_img_size):
    img_w, img_h = img.size
    delta_w = desired_img_size - img_w
    delta_h = desired_img_size - img_h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding, fill="black")
        

def pad_image(img_path, save_dir, desired_img_size):
    img = Image.open(img_path)
    img_w, img_h = img.size
    delta_w = desired_img_size - img_w
    delta_h = desired_img_size - img_h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    padded_img = ImageOps.expand(img, padding, fill="black")
    img_name = img_path.split('/')[-1]
    padded_img_fn = f'{save_dir}/{img_name}'
    padded_img.save(padded_img_fn)


def add_padding_to_images_in_dir(desired_img_size, image_dir, save_dir):
    paths_in_dir = get_file_paths_in_dir(image_dir)
    paths_in_dir = [path for path in paths_in_dir if path.endswith(JPG_EXT)]
    pool_iterable = [(path, save_dir, desired_img_size) for path in paths_in_dir]
    # Parallelize process:
    with Pool(NUM_CPUS) as p:
        p.starmap(pad_image, pool_iterable)


def preprocess_images(directory_list):
    l_width, l_height = get_largest_image_dimensions_in_directories(directory_list)
    d_size = l_width if l_width >= l_height else l_height
    for directory in directory_list:
        preprocessed_dir = create_preprocessed_image_dir(directory)
        add_padding_to_images_in_dir(d_size, directory, preprocessed_dir)


def resize_image(img_path, save_dir, size):
    with Image.open(img_path) as img:
        img.thumbnail((size, size))
        img_name = img_path.split('/')[-1]
        resized_img_fn = f'{save_dir}/{img_name}'
        img.save(resized_img_fn)


# Should only be used to resize square images or else the aspect ratio will change:
def resize_images(source_dir, save_dir, size):
    paths_in_dir = get_file_paths_in_dir(source_dir)
    paths_in_dir = [path for path in paths_in_dir if path.endswith(JPG_EXT)]
    pool_iterable = [(path, save_dir, size) for path in paths_in_dir]
    # Parallelize process:
    with Pool(NUM_CPUS) as p:
        p.starmap(resize_image, pool_iterable)


def add_image_label(img, label):
    return label, img


def add_labels_to_images(img_list, label):
    return [add_image_label(img, label) for img in img_list]


def add_labels_to_images_threaded(img_list, label):
    with ThreadPool(NUM_THREADS) as tp:
        return tp.starmap(add_image_label, [(img, label) for img in img_list])


def add_labels_to_images_generator(img_list, label):
    return (add_image_label(img, label) for img in img_list)


# Naive loading of all images in a directory:
def load_images_in_dir(directory, img_type=JPG_EXT):
    images = []
    for fn in os.listdir(directory):
        if fn.endswith(img_type):
            img = Image.open(f'{directory}/{fn}')
            images.append(img)
    return images


def load_random_image_in_dir(directory, img_type=JPG_EXT):
    images = []
    for fn in os.listdir(directory):
        if fn.endswith(img_type):
            img = Image.open(f'{directory}/{fn}')
            images.append(img)
    return random.choice(images)


def load_images_in_dir_threaded(directory, img_type=JPG_EXT):
    image_filenames = [f'{directory}/{f}' for f in os.listdir(directory) if f.endswith(img_type)]
    # print(f"[load_images_in_dir_threaded] Example image filename: {image_filenames[0]}")
    with ThreadPool(NUM_THREADS) as p:
        return p.map(Image.open, image_filenames)


def load_images_in_dir_generator(directory, img_type=JPG_EXT, shuffle=True, recursive=False):
    if recursive:
        list_dir = []
        for folder,_,fnames in os.walk(directory):
            list_dir.extend(f'{folder}/{fnames}')
    else:
        list_dir = os.listdir(directory)

    if shuffle:
        random.shuffle(list_dir)

    for fn in list_dir:
        if fn.endswith(img_type):
            img = Image.open(f'{directory}/{fn}')
            yield img


def load_images_in_dir_rand_generator(directory, img_type=JPG_EXT):
    dirlist = os.listdir(directory)
    for _ in range(len(dirlist)):
        fn = random.choice(dirlist)
        if fn.endswith(img_type):
            img = Image.open(f'{directory}/{fn}')
            yield img


def normalize_pixels_in_img_obj(image_obj):
    pixels = np.asarray(image_obj)
    pixels = pixels.astype('float32')
    pixels /= 255
    return pixels


def normalize_pixels_in_img_fn(image_fn):
    img = Image.open(image_fn)
    return normalize_pixels_in_img_obj(img)


def load_normalized_images_in_dir(directory, img_type=JPG_EXT):
    normalized_imgs = []
    load_img_generator = load_images_in_dir_generator(directory, img_type)
    for img in load_img_generator:
        normalized = normalize_pixels_in_img_obj(img)
        normalized_imgs.append(normalized)
    return normalized_imgs


def load_random_normalized_image_in_dir(directory, img_type=JPG_EXT):
    img = load_random_image_in_dir(directory, img_type)
    normalized = normalize_pixels_in_img_obj(img)
    return normalized


def load_normalized_images_in_dir_threaded(directory, img_type=JPG_EXT):
    imgs = load_images_in_dir_threaded(directory, img_type)
    with ThreadPool(NUM_THREADS) as p:
        return p.map(normalize_pixels_in_img_obj, imgs)


def load_normalized_images_in_dir_generator(directory, img_type=JPG_EXT):
    load_img_generator = load_images_in_dir_generator(directory, img_type)
    for img in load_img_generator:
        normalized = normalize_pixels_in_img_obj(img)
        yield normalized


def load_normalized_rand_images_in_dir_generator(directory, img_type=JPG_EXT):
    load_img_generator = load_images_in_dir_rand_generator(directory, img_type)
    for img in load_img_generator:
        normalized = normalize_pixels_in_img_obj(img)
        yield normalized


def load_random_cropped_normalized_images_in_dir_generator(directory, crop_size, img_type=JPG_EXT, recursive=True):
    load_img_generator = load_images_in_dir_generator(directory, img_type, recursive=recursive)
    for img in load_img_generator:
        normalized = normalize_pixels_in_img_obj(img)
        yield tf.image.random_crop(normalized, size=crop_size)


def convert_labels_to_one_hot_vectors(labels, encoder=None):
    label_encoder = LabelEncoder() if encoder is None else encoder
    # print(f'label_encoder params: {label_encoder.get_params()}')
    encoder_fit = label_encoder.fit_transform(labels)
    # print(f'encoder_fit: {encoder_fit}')
    integer_encoded_labels = np.array(encoder_fit)
    # print(f'integer_encoded_labels: {integer_encoded_labels}')
    return to_categorical(integer_encoded_labels), label_encoder


def save_label_encoder(encoder, filename, base_dir='models'):
    label_encoder_path = f'{PROJECT_ROOT}/{base_dir}/{filename}'
    np.save(label_encoder_path, encoder.classes_)


def load_label_encoder(filename, base_dir='models'):
    label_encoder_path = f'{PROJECT_ROOT}/{base_dir}/{filename}'
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path)
    return label_encoder
