import os
import shutil
from pathlib import Path
from random import random
from ImagePreprocessing.ImagePreprocessing import get_file_paths_in_dir


TRAIN_FOLDER_NAME = 'train'
VALIDATION_FOLDER_NAME = 'validation'
TEST_FOLDER_NAME = 'test'

DEFAULT_SEED = 42


def create_image_dir(base_dir, dir_name):
    img_dir_name = f'{base_dir}/{dir_name}'
    if not os.path.exists(img_dir_name):
        os.mkdir(img_dir_name)
    return img_dir_name


def create_train_img_dir(base_dir):
    return create_image_dir(base_dir, TRAIN_FOLDER_NAME)


def create_validation_img_dir(base_dir):
    return create_image_dir(base_dir, VALIDATION_FOLDER_NAME)


def create_test_img_dir(base_dir):
    return create_image_dir(base_dir, TEST_FOLDER_NAME)


def copy_images_to_train_validation_test(base_dir, source_dir, validation_pct=0.1, test_pct=0.1):
    train_dir = create_train_img_dir(base_dir)
    valid_dir = create_validation_img_dir(base_dir)
    test_dir = create_test_img_dir(base_dir)

    # Do test-train split, then copy the images over
    train_pct = 1.0 - validation_pct - test_pct
    all_img_file_paths = get_file_paths_in_dir(source_dir)
    for img_file_path in all_img_file_paths:
        print(f'img_file_path: {img_file_path}')
        randf = random()
        img_name = Path(img_file_path).name
        if randf < train_pct:
            dest_fp = f'{base_dir}/{TRAIN_FOLDER_NAME}/{img_name}'
        elif randf < 1.0 - test_pct:
            dest_fp = f'{base_dir}/{VALIDATION_FOLDER_NAME}/{img_name}'
        else:
            dest_fp = f'{base_dir}/{TEST_FOLDER_NAME}/{img_name}'

        shutil.copyfile(img_file_path, dest_fp)

    return train_dir, valid_dir, test_dir
