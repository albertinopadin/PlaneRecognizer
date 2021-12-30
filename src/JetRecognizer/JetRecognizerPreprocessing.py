from Common.DL_FilePaths import FIGHTER_JET, IMG_SIZE, get_fighter_jet_dir
from ImagePreprocessing.ImagePreprocessing import *
from ImagePreprocessing.TrainTestSeparator import copy_images_to_train_validation_test, \
    TRAIN_FOLDER_NAME, VALIDATION_FOLDER_NAME, TEST_FOLDER_NAME
from Common.TimerDecorator import timer
import numpy as np
from enum import Enum


class Label(Enum):
    F22 = 'F-22'
    F14 = 'F-14'
    F15 = 'F-15'
    F16 = 'F-16'
    F18 = 'F-18'
    F35 = 'F-35'


@timer
def preprocess_all_images(size=IMG_SIZE.IS_960):
    img_directories = list()
    for fighter in FIGHTER_JET:
        fdir = get_fighter_jet_dir(fighter, size=size)
        img_directories.append(fdir)
    preprocess_images(img_directories)


def get_f22_preprocessed_images(size=IMG_SIZE.IS_960):
    f22_preprocessed_folder = f'{get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)}/{PREPROCESSED_FOLDER_NAME}'
    image_generator = load_images_in_dir_generator(f22_preprocessed_folder)
    return image_generator


# def get_f22_images_as_pixel_values():
#     print('Getting F-22 preprocessed images...')
#     pp_start = time.time()
#     image_generator = get_f22_preprocessed_images()
#     pp_end = time.time()
#     print(f'Took {pp_end - pp_start} time to get preprocessed images.')
#
#     print('Normalizing pixels in F-22 images...')
#     np_start = time.time()
#     normalized_images = [normalize_pixels_in_img_obj(img) for img in image_generator]
#     np_end = time.time()
#     print(f'Took {np_end - np_start} time to get normalized pixel images.')
#     return normalized_images


# --------------------- LOOP METHODS --------------------- #
@timer
def get_f22_preprocessed_images_as_pixel_values(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)
    normalized_imgs = load_normalized_images_in_dir(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images(normalized_imgs, Label.F22.value)


# TODO: Can't do a batch load like this because need to attach labels:
# @timer
# def get_non_f22_preprocessed_images_as_pixel_values():
#     return [load_normalized_images_in_dir(f'{folder}/{PREPROCESSED_FOLDER_NAME}') for folder in NON_F22_IMAGE_FOLDERS]


@timer
def get_f14_preprocessed_images_as_pixel_values(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_14, size=size)
    normalized_imgs = load_normalized_images_in_dir(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images(normalized_imgs, Label.F14.value)


@timer
def get_f15_preprocessed_images_as_pixel_values(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_15, size=size)
    normalized_imgs = load_normalized_images_in_dir(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images(normalized_imgs, Label.F15.value)


@timer
def get_f16_preprocessed_images_as_pixel_values(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    normalized_imgs = load_normalized_images_in_dir(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images(normalized_imgs, Label.F16.value)


@timer
def get_f18_preprocessed_images_as_pixel_values(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    normalized_imgs = load_normalized_images_in_dir(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images(normalized_imgs, Label.F18.value)


@timer
def get_f35_preprocessed_images_as_pixel_values(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_35, size=size)
    normalized_imgs = load_normalized_images_in_dir(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images(normalized_imgs, Label.F35.value)


# --------------------- THREADED METHODS --------------------- #
@timer
def get_f22_preprocessed_images_as_pixel_values_threaded(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)
    normalized_imgs = load_normalized_images_in_dir_threaded(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images_threaded(normalized_imgs, Label.F22.value)


@timer
def get_f14_preprocessed_images_as_pixel_values_threaded(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_14, size=size)
    normalized_imgs = load_normalized_images_in_dir_threaded(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images_threaded(normalized_imgs, Label.F14.value)


@timer
def get_f15_preprocessed_images_as_pixel_values_threaded(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_15, size=size)
    normalized_imgs = load_normalized_images_in_dir_threaded(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images_threaded(normalized_imgs, Label.F15.value)


@timer
def get_f16_preprocessed_images_as_pixel_values_threaded(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    normalized_imgs = load_normalized_images_in_dir_threaded(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images_threaded(normalized_imgs, Label.F16.value)


@timer
def get_f18_preprocessed_images_as_pixel_values_threaded(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_18, size=size)
    normalized_imgs = load_normalized_images_in_dir_threaded(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images_threaded(normalized_imgs, Label.F18.value)


@timer
def get_f35_preprocessed_images_as_pixel_values_threaded(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_35, size=size)
    normalized_imgs = load_normalized_images_in_dir_threaded(f'{jet_dir}/{PREPROCESSED_FOLDER_NAME}')
    return add_labels_to_images_threaded(normalized_imgs, Label.F35.value)


# --------------------- GENERATOR METHODS --------------------- #
def get_random_normalized_images_generator(jet_dir, folder_name, label):
    normalized_img_generator = load_normalized_rand_images_in_dir_generator(f'{jet_dir}/{folder_name}')
    return add_labels_to_images_generator(normalized_img_generator, label.value)


def get_f22_preprocessed_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)
    print(f'Getting F-22 preprocessed images from: {jet_dir}')
    return get_random_normalized_images_generator(jet_dir, PREPROCESSED_FOLDER_NAME, Label.F22)


def get_f14_preprocessed_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_14, size=size)
    return get_random_normalized_images_generator(jet_dir, PREPROCESSED_FOLDER_NAME, Label.F14)


def get_f15_preprocessed_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_15, size=size)
    return get_random_normalized_images_generator(jet_dir, PREPROCESSED_FOLDER_NAME, Label.F15)


def get_f16_preprocessed_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    return get_random_normalized_images_generator(jet_dir, PREPROCESSED_FOLDER_NAME, Label.F16)


def get_f18_preprocessed_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_18, size=size)
    return get_random_normalized_images_generator(jet_dir, PREPROCESSED_FOLDER_NAME, Label.F18)


def get_f35_preprocessed_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_35, size=size)
    return get_random_normalized_images_generator(jet_dir, PREPROCESSED_FOLDER_NAME, Label.F35)


def get_random_preprocessed_fighter_images_as_pixel_values_generator(num_batches=1):
    f22_gen = get_f22_preprocessed_images_as_pixel_values_generator()
    f14_gen = get_f14_preprocessed_images_as_pixel_values_generator()
    f15_gen = get_f15_preprocessed_images_as_pixel_values_generator()
    f16_gen = get_f16_preprocessed_images_as_pixel_values_generator()
    f18_gen = get_f18_preprocessed_images_as_pixel_values_generator()
    f35_gen = get_f35_preprocessed_images_as_pixel_values_generator()

    fighter_img_generator_list = [
        f22_gen,
        f14_gen,
        f15_gen,
        f16_gen,
        f18_gen,
        f35_gen
    ]

    while True:
        img_batch_list = []
        for _ in range(num_batches):
            img_list = [next(gen) for gen in fighter_img_generator_list]
            img_list = [img for img in img_list if img is not None]
            img_batch_list.extend(img_list)

        if len(img_batch_list) == 0:
            # TODO: Perhaps raise StopIteration here?
            break

        np.random.shuffle(img_batch_list)
        yield img_batch_list


# --------------------- TRAIN --------------------- #
def get_f22_train_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)
    print(f'Getting F-22 preprocessed images from: {jet_dir}')
    return get_random_normalized_images_generator(jet_dir, TRAIN_FOLDER_NAME, Label.F22)


def get_f14_train_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_14, size=size)
    return get_random_normalized_images_generator(jet_dir, TRAIN_FOLDER_NAME, Label.F14)


def get_f15_train_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_15, size=size)
    return get_random_normalized_images_generator(jet_dir, TRAIN_FOLDER_NAME, Label.F15)


def get_f16_train_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    return get_random_normalized_images_generator(jet_dir, TRAIN_FOLDER_NAME, Label.F16)


def get_f18_train_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_18, size=size)
    return get_random_normalized_images_generator(jet_dir, TRAIN_FOLDER_NAME, Label.F18)


def get_f35_train_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_35, size=size)
    return get_random_normalized_images_generator(jet_dir, TRAIN_FOLDER_NAME, Label.F35)


# --------------------- VALIDATION --------------------- #
def get_f22_validation_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)
    return get_random_normalized_images_generator(jet_dir, VALIDATION_FOLDER_NAME, Label.F22)


def get_f14_validation_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_14, size=size)
    return get_random_normalized_images_generator(jet_dir, VALIDATION_FOLDER_NAME, Label.F14)


def get_f15_validation_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_15, size=size)
    return get_random_normalized_images_generator(jet_dir, VALIDATION_FOLDER_NAME, Label.F15)


def get_f16_validation_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    return get_random_normalized_images_generator(jet_dir, VALIDATION_FOLDER_NAME, Label.F16)


def get_f18_validation_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_18, size=size)
    return get_random_normalized_images_generator(jet_dir, VALIDATION_FOLDER_NAME, Label.F18)


def get_f35_validation_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_35, size=size)
    return get_random_normalized_images_generator(jet_dir, VALIDATION_FOLDER_NAME, Label.F35)


# --------------------- TEST --------------------- #
def get_f22_test_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_22, size=size)
    print(f'Getting F-22 preprocessed images from: {jet_dir}')
    return get_random_normalized_images_generator(jet_dir, TEST_FOLDER_NAME, Label.F22)


def get_f14_test_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_14, size=size)
    return get_random_normalized_images_generator(jet_dir, TEST_FOLDER_NAME, Label.F14)


def get_f15_test_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_15, size=size)
    return get_random_normalized_images_generator(jet_dir, TEST_FOLDER_NAME, Label.F15)


def get_f16_test_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_16, size=size)
    return get_random_normalized_images_generator(jet_dir, TEST_FOLDER_NAME, Label.F16)


def get_f18_test_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_18, size=size)
    return get_random_normalized_images_generator(jet_dir, TEST_FOLDER_NAME, Label.F18)


def get_f35_test_images_as_pixel_values_generator(size=IMG_SIZE.IS_960):
    jet_dir = get_fighter_jet_dir(FIGHTER_JET.F_35, size=size)
    return get_random_normalized_images_generator(jet_dir, TEST_FOLDER_NAME, Label.F35)


@timer
def resize_fighter_images(from_size=IMG_SIZE.IS_1920, to_size=IMG_SIZE.IS_960):
    for fighter in FIGHTER_JET:
        from_dir = get_fighter_jet_dir(fighter, size=from_size)
        resized_save_dir = get_fighter_jet_dir(fighter, size=to_size) + '/resized'
        if not os.path.exists(resized_save_dir):
            os.makedirs(resized_save_dir)
        resize_images(source_dir=from_dir, save_dir=resized_save_dir, size=to_size.value)

        preprocessed_save_dir = get_fighter_jet_dir(fighter, size=to_size) + '/preprocessed'
        if not os.path.exists(preprocessed_save_dir):
            os.makedirs(preprocessed_save_dir)
        if to_size == IMG_SIZE.IS_960:
            add_padding_to_images_in_dir(960, resized_save_dir, preprocessed_save_dir)


def get_all_fighter_img_dirs(size=IMG_SIZE.IS_960):
    img_directories = []
    for fighter in FIGHTER_JET:
        fdir = get_fighter_jet_dir(fighter, size=size)
        img_directories.append(fdir)
    return img_directories


def create_train_validation_test_split():
    train_valid_test_dirs = []
    img_dirs = get_all_fighter_img_dirs()
    for img_dir in img_dirs:
        train_dir, valid_dir, test_dir = copy_images_to_train_validation_test(base_dir=img_dir,
                                                                              source_dir=img_dir+f'/{PREPROCESSED_FOLDER_NAME}')
        train_valid_test_dirs.append((train_dir, valid_dir, test_dir))
    return train_valid_test_dirs


def shuffle_and_return_generator(num_batches, generator_list):
    while True:
        img_batch_list = []
        for _ in range(num_batches):
            img_list = [next(gen) for gen in generator_list]
            img_list = [img for img in img_list if img is not None]
            img_batch_list.extend(img_list)

        if len(img_batch_list) == 0:
            # TODO: Perhaps raise StopIteration here?
            break

        np.random.shuffle(img_batch_list)
        yield img_batch_list


def get_random_train_fighter_images_as_pixel_values_generator(num_batches=1):
    f22_gen = get_f22_train_images_as_pixel_values_generator()
    f14_gen = get_f14_train_images_as_pixel_values_generator()
    f15_gen = get_f15_train_images_as_pixel_values_generator()
    f16_gen = get_f16_train_images_as_pixel_values_generator()
    f18_gen = get_f18_train_images_as_pixel_values_generator()
    f35_gen = get_f35_train_images_as_pixel_values_generator()

    fighter_img_generator_list = [
        f22_gen,
        f14_gen,
        f15_gen,
        f16_gen,
        f18_gen,
        f35_gen
    ]

    return shuffle_and_return_generator(num_batches, fighter_img_generator_list)


def get_random_validation_fighter_images_as_pixel_values_generator(num_batches=1):
    f22_gen = get_f22_validation_images_as_pixel_values_generator()
    f14_gen = get_f14_validation_images_as_pixel_values_generator()
    f15_gen = get_f15_validation_images_as_pixel_values_generator()
    f16_gen = get_f16_validation_images_as_pixel_values_generator()
    f18_gen = get_f18_validation_images_as_pixel_values_generator()
    f35_gen = get_f35_validation_images_as_pixel_values_generator()

    fighter_img_generator_list = [
        f22_gen,
        f14_gen,
        f15_gen,
        f16_gen,
        f18_gen,
        f35_gen
    ]

    return shuffle_and_return_generator(num_batches, fighter_img_generator_list)


def get_random_test_fighter_images_as_pixel_values_generator(num_batches=1):
    f22_gen = get_f22_test_images_as_pixel_values_generator()
    f14_gen = get_f14_test_images_as_pixel_values_generator()
    f15_gen = get_f15_test_images_as_pixel_values_generator()
    f16_gen = get_f16_test_images_as_pixel_values_generator()
    f18_gen = get_f18_test_images_as_pixel_values_generator()
    f35_gen = get_f35_test_images_as_pixel_values_generator()

    fighter_img_generator_list = [
        f22_gen,
        f14_gen,
        f15_gen,
        f16_gen,
        f18_gen,
        f35_gen
    ]

    return shuffle_and_return_generator(num_batches, fighter_img_generator_list)