import random
import tensorflow as tf
from JetRecognizer.ImageGenerator import get_img_paths_and_labels


class ImageDatasetLoader:
    def __init__(self, img_dir, crop_size=(862,862,3), batch_size=2, label_encoder=None, one_hot_labels=False, validation=False):
        if one_hot_labels:
            if label_encoder is None:
                raise Exception('Must set label encoder to one-hot encode labels.')

        self.img_dir = img_dir
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.label_encoder = label_encoder
        self.one_hot_labels = one_hot_labels
        img_paths, labels = get_img_paths_and_labels(img_dir, one_hot_encode=one_hot_labels, label_encoder=label_encoder)
        self.dataset = ImageDatasetLoader.create_dataset(img_paths, labels, batch_size, crop_size, validation=validation)

    @staticmethod
    def random_flips(img_tensor, label):
        img_tensor = tf.image.random_flip_left_right(img_tensor, seed=random.randint(0, 100))
        img_tensor = tf.image.random_flip_up_down(img_tensor, seed=random.randint(0, 100))
        return img_tensor, label

    @staticmethod
    def random_crop(crop_size):
        def rand_crop(img_tensor, label):
            def _random_crop(_img_tensor):
                x, y, _ = _img_tensor.shape
                crop_width, crop_height, _ = crop_size
                if x < crop_width or y < crop_height:
                    cropped = tf.image.resize_with_crop_or_pad(_img_tensor, crop_width, crop_height)
                else:
                    cropped = tf.image.random_crop(_img_tensor, size=crop_size)
                return cropped

            img_tensor_shape = img_tensor.shape
            [img_tensor, ] = tf.py_function(_random_crop, [img_tensor], [tf.float32])
            img_tensor.set_shape(img_tensor_shape)
            return img_tensor, label
        return rand_crop

    @staticmethod
    def center_crop(crop_size):
        def c_crop(img_tensor, label):
            def _center_crop(_img_tensor):
                crop_width, crop_height, _ = crop_size
                cropped = tf.image.resize_with_crop_or_pad(_img_tensor, crop_width, crop_height)
                return cropped

            img_tensor_shape = img_tensor.shape
            [img_tensor, ] = tf.py_function(_center_crop, [img_tensor], [tf.float32])
            img_tensor.set_shape(img_tensor_shape)
            return img_tensor, label
        return c_crop

    @staticmethod
    def load_image_normalized(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    @staticmethod
    def create_dataset(img_paths, labels, batch_size, crop_size, validation=False):
        img_paths_t = tf.constant(img_paths)
        labels_t = tf.constant(labels)
        ds = tf.data.Dataset.from_tensor_slices((img_paths_t, labels_t))
        ds = ds.map(
            ImageDatasetLoader.load_image_normalized,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).map(
            ImageDatasetLoader.random_flips,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if validation:
            ds = ds.map(
                ImageDatasetLoader.center_crop(crop_size),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        else:
            ds = ds.map(
                ImageDatasetLoader.random_crop(crop_size),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
