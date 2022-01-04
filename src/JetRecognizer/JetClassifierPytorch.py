from ImagePreprocessing.ImagePreprocessing import load_label_encoder, save_label_encoder, \
    convert_labels_to_one_hot_vectors
from JetRecognizerPreprocessing import get_random_train_fighter_images_as_pixel_values_generator, \
    get_random_validation_fighter_images_as_pixel_values_generator
from BatchLoopGenerator import image_batch_loop
import numpy as np
from time import perf_counter
from Common.Platforms import in_mac_os
from ImageObjectDetectors.TorchCNN4Images import TorchCNN4Images
from sklearn.metrics import precision_score, accuracy_score

# INPUT_SHAPE = (1920, 1920, 3)
INPUT_SHAPE = (960, 960, 3)  # Trying smaller images
N_OUTPUT = 6
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.03 if in_mac_os() else 0.03  # Best learning rate

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

jet_recognizer = TorchCNN4Images(INPUT_SHAPE, N_OUTPUT, LEARNING_RATE)

if LOAD_EXISTING_MODEL:
    jet_recognizer.load_model(JET_RECOGNIZER_MODEL_FILENAME)

if LOAD_EXISTING_LABEL_ENCODER:
    label_encoder = load_label_encoder(LABEL_ENCODER_FILENAME)
else:
    label_encoder = None

BATCH_LOOPS = 10 if in_mac_os() else 25
NUM_GEN_BATCHES = 30 if in_mac_os() else 30

train_random_img_batch_generator = \
    get_random_train_fighter_images_as_pixel_values_generator(num_batches=NUM_GEN_BATCHES)

batch_size = 8 if in_mac_os() else 8  # Getting error in macOS if I use batch size > 2 for 1920x1920 images
n_epochs = 4 if in_mac_os() else 4
train_validation_split = 0.1

print(
    f"\n************ Starting training for {BATCH_LOOPS} batch loops in {'macOS' if in_mac_os() else 'Linux'}... ************\n")
_start = perf_counter()
for train_images, train_labels in image_batch_loop(BATCH_LOOPS, train_random_img_batch_generator):
    print(f'Train labels (before one-hot conversion), len: {len(train_labels)}, : {train_labels}')
    train_labels, label_encoder = convert_labels_to_one_hot_vectors(train_labels, encoder=label_encoder)
    # TODO: How to get validation set if we are using a generator? Probably need to look into the fit_generator methods
    jet_recognizer.train(train_images=train_images,
                         train_labels=train_labels,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         train_validation_split=train_validation_split)
_end = perf_counter()
_elapsed = _end - _start
print(
    f"\n************ Training {BATCH_LOOPS} loops took {int(_elapsed / 60)} minutes {int(_elapsed % 60)} seconds). ************\n")

if SAVE_MODEL:
    jet_recognizer.save_model(JET_RECOGNIZER_MODEL_FILENAME)

if SAVE_LABEL_ENCODER:
    save_label_encoder(label_encoder, LABEL_ENCODER_FILENAME)

NUM_VAL_BATCHES = 8
validation_random_img_batch_generator = get_random_validation_fighter_images_as_pixel_values_generator(
    num_batches=NUM_VAL_BATCHES)
small_validation_sample_list = next(validation_random_img_batch_generator)
validation_images = np.array([img for label, img in small_validation_sample_list])
validation_labels = [label for label, img in small_validation_sample_list]
validation_labels, _ = convert_labels_to_one_hot_vectors(validation_labels, encoder=label_encoder)
predictions = jet_recognizer.predict(validation_images, flatten_output=False, one_hot=True)

print_individual = False
if print_individual:
    print("Predictions vs. Ground Truth:")
    for t_pred, t_label in zip(predictions, validation_labels):
        print(f'Prediction: {t_pred}, Truth: {t_label}')

print(f"Predictions type: {type(predictions)}")
print(f"Validation labels type: {type(validation_labels)}")
print(f"Model accuracy in validation set: {accuracy_score(validation_labels, predictions):0.4f}")
