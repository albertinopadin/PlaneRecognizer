import tensorflow as tf
from ImagePreprocessing.ImagePreprocessing import load_random_cropped_normalized_images_in_dir_generator

# TODO: Images don't have labels, need to add that!
class ImageIterator:
    def __init__(self, img_dir, crop_size=(862,862), batch_size=2):
        self.batch_size = batch_size
        self.generator = load_random_cropped_normalized_images_in_dir_generator(img_dir, crop_size=crop_size)

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(next(self.generator))
            except StopIteration as e:
                print('Caught StopIteration')
                break

        for i in range(len(batch)):
            batch[i] = tf.image.random_flip_left_right(batch[i])
            batch[i] = tf.image.random_flip_up_down(batch[i])

        return batch

