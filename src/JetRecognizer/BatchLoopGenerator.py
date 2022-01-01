import numpy as np


def image_batch_loop(num_batches, image_generator):
    for i in range(num_batches):
        print(f"Batch Loop #{i+1}")
        try:
            small_training_sample_list = next(image_generator)
            train_images = np.asarray([img for label, img in small_training_sample_list])
            print(f'Train images len: {len(train_images)}')
            train_labels = [label for label, img in small_training_sample_list]
            del small_training_sample_list
            yield train_images, train_labels
        except StopIteration as e:
            print(f"No more images for batch {i+1}; {e}")
            break
