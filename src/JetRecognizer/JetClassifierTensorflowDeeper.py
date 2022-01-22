from ImagePreprocessing.ImagePreprocessing import load_label_encoder, save_label_encoder, \
    convert_labels_to_one_hot_vectors
from JetRecognizerPreprocessing import get_random_train_fighter_images_as_pixel_values_generator, \
    get_random_validation_fighter_images_as_pixel_values_generator
from BatchLoopGenerator import image_batch_loop
import numpy as np
from time import perf_counter
from Common.Platforms import in_mac_os
from ImageObjectDetectors.TensorflowDeeperCNN import TensorflowDeeperCNN
from sklearn.metrics import accuracy_score
from ClassifierTestUtils import show_tensorflow_history, show_tensorflow_histories
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Common.DL_FilePaths import SIZE_960_DIR, SIZE_1920_DIR
from ImageGenerator import ImageGenerator
from ImageDatasetLoader import ImageDatasetLoader


# tf.debugging.set_log_device_placement(True)

INPUT_SHAPE = (862, 862, 3)

if in_mac_os():
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_apple_silicon_' + str(INPUT_SHAPE[0])
    print('In macOS!')
else:
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_A6000_' + str(INPUT_SHAPE[0])

N_OUTPUT = 6

# LEARNING_RATE = 0.1 if in_mac_os() else 0.1
# LEARNING_RATE = 0.03 if in_mac_os() else 0.03
# LEARNING_RATE = 0.01 if in_mac_os() else 0.01  # Best starting learning rate
# LEARNING_RATE = 0.003 if in_mac_os() else 0.003
# LEARNING_RATE = 0.001 if in_mac_os() else 0.001
# LEARNING_RATE = 0.0003 if in_mac_os() else 0.0003
LEARNING_RATE = 0.0001 if in_mac_os() else 0.0001  # Good starting for Adam
# LEARNING_RATE = 0.00001 if in_mac_os() else 0.00001
# LEARNING_RATE = 0.000001 if in_mac_os() else 0.000001
# LEARNING_RATE = 0.0000001 if in_mac_os() else 0.0000001

DROPOUT = 0.5
OPTIM = 'adam'
ACTIVATION = 'mish'

# Set the following flag to load a saved model:
LOAD_EXISTING_MODEL = True if in_mac_os() else False
# Set the following flag to save model after training:
SAVE_MODEL = True

LABEL_ENCODER_FILENAME = "jet_label_classes.npy"
# Set to load a saved label encoder:
LOAD_EXISTING_LABEL_ENCODER = True
# Set to save the label encoder:
SAVE_LABEL_ENCODER = True if in_mac_os() else True

USING_CHECKPOINTS = True

jet_recognizer = TensorflowDeeperCNN(INPUT_SHAPE,
                                     N_OUTPUT,
                                     LEARNING_RATE,
                                     activation=ACTIVATION,
                                     dropout=DROPOUT,
                                     optimizer=OPTIM,
                                     filename=JET_RECOGNIZER_MODEL_FILENAME)

if LOAD_EXISTING_MODEL:
    jet_recognizer.load_model(JET_RECOGNIZER_MODEL_FILENAME, is_checkpoint=USING_CHECKPOINTS)

if LOAD_EXISTING_LABEL_ENCODER:
    label_encoder = load_label_encoder(LABEL_ENCODER_FILENAME)
else:
    label_encoder = None

# BATCH_LOOPS = 25 if in_mac_os() else 25
# NUM_GEN_BATCHES = 30 if in_mac_os() else 30
#
# train_random_img_batch_generator = get_random_train_fighter_images_as_pixel_values_generator(
#     num_batches=NUM_GEN_BATCHES)
#
# batch_size = 2 if in_mac_os() else 2
# n_epochs = 3 if in_mac_os() else 3
# train_validation_split = 0.1
#
# print(
#     f"\n************ Starting training for {BATCH_LOOPS} batch loops in {'macOS' if in_mac_os() else 'Linux'}... ************\n")
# _start = perf_counter()
# histories = []
# for train_images, train_labels in image_batch_loop(BATCH_LOOPS, train_random_img_batch_generator):
#     print(f'Train labels (before one-hot conversion), len: {len(train_labels)}, : {train_labels}')
#     train_labels, label_encoder = convert_labels_to_one_hot_vectors(train_labels, encoder=label_encoder)
#     # TODO: How to get validation set if we are using a generator? Probably need to look into the fit_generator methods
#     history = jet_recognizer.train(train_images=train_images,
#                                    train_labels=train_labels,
#                                    batch_size=batch_size,
#                                    n_epochs=n_epochs,
#                                    train_validation_split=train_validation_split)
#     histories.append(history)
# _end = perf_counter()
# _elapsed = _end - _start
# print(
#     f"\n************ Training {BATCH_LOOPS} loops took {int(_elapsed / 60)} minutes {int(_elapsed % 60)} seconds). ************\n")
#
# show_tensorflow_histories(histories)

img_target_size = (862, 862, 3)
batch_size = 8 if in_mac_os() else 8

train_dir = SIZE_1920_DIR + '/train'
# train_gen = ImageGenerator(train_dir,
#                            crop_size=img_target_size,
#                            batch_size=batch_size,
#                            label_encoder=label_encoder,
#                            one_hot_labels=True)
#
# valid_dir = SIZE_1920_DIR + '/validation'
# valid_gen = ImageGenerator(valid_dir,
#                            crop_size=img_target_size,
#                            batch_size=batch_size,
#                            label_encoder=label_encoder,
#                            one_hot_labels=True)

train_dsl = ImageDatasetLoader(train_dir,
                               crop_size=img_target_size,
                               batch_size=batch_size,
                               label_encoder=label_encoder,
                               one_hot_labels=True)

valid_dir = SIZE_1920_DIR + '/validation'
valid_dsl = ImageDatasetLoader(valid_dir,
                               crop_size=img_target_size,
                               batch_size=batch_size,
                               label_encoder=label_encoder,
                               one_hot_labels=True,
                               validation=True)

n_epochs = 3 if in_mac_os() else 30

print(f"\n************ Starting training for {n_epochs} epochs in "
      f"{'macOS' if in_mac_os() else 'Linux'}... ************\n")
_start = perf_counter()
history = jet_recognizer.train_all(train_gen=train_dsl.dataset,
                                   valid_gen=valid_dsl.dataset,
                                   n_epochs=n_epochs,
                                   batch_size=batch_size)
_end = perf_counter()
_elapsed = _end - _start
print(
    f"\n************ Training {n_epochs} epochs took {int(_elapsed / 60)} minutes "
    f"{int(_elapsed % 60)} seconds). ************\n")

if SAVE_MODEL:
    jet_recognizer.save_model(JET_RECOGNIZER_MODEL_FILENAME, using_checkpoints=USING_CHECKPOINTS)

if SAVE_LABEL_ENCODER:
    save_label_encoder(label_encoder, LABEL_ENCODER_FILENAME)

show_tensorflow_history(history)

# NUM_VAL_BATCHES = 8
# validation_random_img_batch_generator = get_random_validation_fighter_images_as_pixel_values_generator(
#     num_batches=NUM_VAL_BATCHES)
# small_validation_sample_list = next(validation_random_img_batch_generator)


def validate_model(jet_classifier, img_dir, target_size, test=False):
    print(f"{'Testing' if test else 'Validating'} model with images from directory: {img_dir} ...")
    test_dsl = ImageDatasetLoader(img_dir,
                                  crop_size=target_size,
                                  batch_size=200,
                                  label_encoder=label_encoder,
                                  one_hot_labels=True,
                                  validation=True)

    validation_images, validation_labels = next(iter(test_dsl.dataset))
    predictions = jet_classifier.predict(validation_images, flatten_output=False, one_hot=True)

    print_individual = False
    if print_individual:
        print("Predictions vs. Ground Truth:")
        for t_pred, t_label in zip(predictions, validation_labels):
            print(f'Prediction: {t_pred}, Truth: {t_label}')

    print(f"Predictions type: {type(predictions)}")
    print(f"{'Test' if test else 'Validation'} labels type: {type(validation_labels)}")
    print(f"Model accuracy in {'test' if test else 'validation'} set: "
          f"{accuracy_score(validation_labels, predictions):0.4f}")


valid_dir = SIZE_1920_DIR + '/validation'
test_dir = SIZE_1920_DIR + '/test'

validate_model(jet_recognizer, valid_dir, img_target_size)
validate_model(jet_recognizer, test_dir, img_target_size, test=True)
