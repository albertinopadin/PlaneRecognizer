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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Common.DL_FilePaths import SIZE_960_DIR, SIZE_1920_DIR


INPUT_SHAPE = (862, 862, 3)
N_OUTPUT = 6
# LEARNING_RATE = 0.1 if in_mac_os() else 0.1
# LEARNING_RATE = 0.03 if in_mac_os() else 0.03
# LEARNING_RATE = 0.01 if in_mac_os() else 0.01  # Best starting learning rate
LEARNING_RATE = 0.003 if in_mac_os() else 0.003
# LEARNING_RATE = 0.001 if in_mac_os() else 0.001
# LEARNING_RATE = 0.0003 if in_mac_os() else 0.0003
# LEARNING_RATE = 0.0001 if in_mac_os() else 0.0001

if in_mac_os():
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_apple_silicon_' + str(INPUT_SHAPE[0])
    print('In macOS!')
else:
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_A6000_' + str(INPUT_SHAPE[0])

# Set the following flag to load a saved model:
LOAD_EXISTING_MODEL = True if in_mac_os() else False
# Set the following flag to save model after training:
SAVE_MODEL = True

LABEL_ENCODER_FILENAME = "jet_label_classes.npy"
# Set to load a saved label encoder:
LOAD_EXISTING_LABEL_ENCODER = True
# Set to save the label encoder:
SAVE_LABEL_ENCODER = True if in_mac_os() else True

jet_recognizer = TensorflowDeeperCNN(INPUT_SHAPE, N_OUTPUT, LEARNING_RATE)

if LOAD_EXISTING_MODEL:
    jet_recognizer.load_model(JET_RECOGNIZER_MODEL_FILENAME)

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

# From: https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7

train_dir = SIZE_1920_DIR + '/train'

augs_gen = ImageDataGenerator(
    data_format='channels_last',
    rescale=1./255,
    horizontal_flip=True,
    # height_shift_range=.2,
    vertical_flip=True,
    validation_split=0.1
)


img_target_size = (862, 862)
batch_size = 2 if in_mac_os() else 2

train_gen = augs_gen.flow_from_directory(
    train_dir,
    target_size=img_target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)

valid_gen = augs_gen.flow_from_directory(
    train_dir,
    target_size=img_target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

n_epochs = 2 if in_mac_os() else 10

print(f"\n************ Starting training for {n_epochs} epochs in "
      f"{'macOS' if in_mac_os() else 'Linux'}... ************\n")
_start = perf_counter()
history = jet_recognizer.train_all(train_gen=train_gen, valid_gen=valid_gen, n_epochs=n_epochs, batch_size=batch_size)
_end = perf_counter()
_elapsed = _end - _start
print(
    f"\n************ Training {n_epochs} epochs took {int(_elapsed / 60)} minutes "
    f"{int(_elapsed % 60)} seconds). ************\n")

show_tensorflow_history(history)

if SAVE_MODEL:
    jet_recognizer.save_model(JET_RECOGNIZER_MODEL_FILENAME)

if SAVE_LABEL_ENCODER:
    save_label_encoder(label_encoder, LABEL_ENCODER_FILENAME)

NUM_VAL_BATCHES = 8
# validation_random_img_batch_generator = get_random_validation_fighter_images_as_pixel_values_generator(
#     num_batches=NUM_VAL_BATCHES)
# small_validation_sample_list = next(validation_random_img_batch_generator)

valid_dir = SIZE_1920_DIR + '/validation'

test_img_gen = ImageDataGenerator(
    data_format='channels_last',
    rescale=1./255
)

test_gen = augs_gen.flow_from_directory(
    valid_dir,
    target_size=img_target_size,
    batch_size=300,
    class_mode='categorical',
    shuffle=False
)

# validation_images = np.array([img for label, img in small_validation_sample_list])
# validation_labels = [label for label, img in small_validation_sample_list]
# validation_labels, _ = convert_labels_to_one_hot_vectors(validation_labels, encoder=label_encoder)

validation_images, validation_labels = next(test_gen)
predictions = jet_recognizer.predict(validation_images, flatten_output=False, one_hot=True)

print_individual = False
if print_individual:
    print("Predictions vs. Ground Truth:")
    for t_pred, t_label in zip(predictions, validation_labels):
        print(f'Prediction: {t_pred}, Truth: {t_label}')

print(f"Predictions type: {type(predictions)}")
print(f"Validation labels type: {type(validation_labels)}")
print(f"Model accuracy in validation set: {accuracy_score(validation_labels, predictions):0.4f}")
