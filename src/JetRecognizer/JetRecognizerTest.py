from ImageObjectDetectors.TensorflowCNN4Images import TensorflowCNN4Images
from JetRecognizerPreprocessing import load_label_encoder, convert_labels_to_one_hot_vectors, \
    get_random_validation_fighter_images_as_pixel_values_generator, \
    get_random_test_fighter_images_as_pixel_values_generator, \
    get_validation_fighter_images_as_pixel_values, get_test_fighter_images_as_pixel_values
import numpy as np
from time import perf_counter
from Common.Platforms import in_mac_os
from ImageObjectDetectors.CNN4ImagesBase import KernelProgression
from sklearn.metrics import precision_score, accuracy_score
from Common.DL_FilePaths import FIGHTER_JET


def test_recognizer(recognizer, sample_list, encoder, test_set=False, print_individual=False):
    print(f"************ Starting inference in {'macOS' if in_mac_os() else 'Linux'}... ************")
    _start = perf_counter()
    test_images = np.array([img for label, img in sample_list])
    test_labels = [label for label, img in sample_list]
    test_labels, _ = convert_labels_to_one_hot_vectors(test_labels, encoder=encoder)
    predictions = recognizer.predict(test_images, flatten_output=False, one_hot=True)
    _end = perf_counter()
    _elapsed = _end - _start
    print(f"************ Inference on {len(test_images)} images took {_elapsed:0.4f} seconds). ************")

    if print_individual:
        print("Predictions vs. Ground Truth:")
        for t_pred, t_label in zip(predictions, test_labels):
            print(f'Prediction: {t_pred}, Truth: {t_label}')

    print(f"Predictions type: {type(predictions)}")
    print(f"{'Test' if test_set else 'Validation'} labels type: {type(test_labels)}")
    print(f"Model accuracy in {'Test' if test_set else 'Validation'}: {accuracy_score(test_labels, predictions):0.4f}\n")


INPUT_SHAPE = (960, 960, 3)
N_OUTPUT = 6
LEARNING_RATE = 0.001

if in_mac_os():
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_apple_silicon' + '_' + str(INPUT_SHAPE[0])
    print('In macOS!')
else:
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_A6000' + '_' + str(INPUT_SHAPE[0])

LABEL_ENCODER_FILENAME = "jet_label_classes.npy"

# Still need to define the model, might want to refactor so don't have to...
jet_recognizer = TensorflowCNN4Images(INPUT_SHAPE,
                                      N_OUTPUT,
                                      LEARNING_RATE,
                                      kernel_progression=KernelProgression.KERNEL_GETS_BIGGER)

jet_recognizer.load_model(JET_RECOGNIZER_MODEL_FILENAME)
label_encoder = load_label_encoder(LABEL_ENCODER_FILENAME)

ONLY_SAMPLE = False
# JET_TYPES = None
# JET_TYPES = [t for t in FIGHTER_JET]
# Model is least accurate with F-18
JET_TYPES = [
    # FIGHTER_JET.F_14,
    # FIGHTER_JET.F_15,
    FIGHTER_JET.F_16,
    # FIGHTER_JET.F_18,
    FIGHTER_JET.F_22,
    FIGHTER_JET.F_35
]

print(f"Jet types: {JET_TYPES}")

if ONLY_SAMPLE:
    NUM_VALID_BATCHES = 60
    validation_random_img_batch_generator = get_random_validation_fighter_images_as_pixel_values_generator(num_batches=NUM_VALID_BATCHES)
    validation_sample_list = next(validation_random_img_batch_generator)
else:
    print('Getting validation samples...')
    validation_sample_list = get_validation_fighter_images_as_pixel_values(jet_types=JET_TYPES)

print("\n************ VALIDATION ************")
test_recognizer(jet_recognizer, validation_sample_list, label_encoder)

if ONLY_SAMPLE:
    NUM_TEST_BATCHES = 60
    test_random_img_batch_generator = get_random_test_fighter_images_as_pixel_values_generator(num_batches=NUM_TEST_BATCHES)
    test_sample_list = next(test_random_img_batch_generator)
else:
    print('Getting test samples...')
    test_sample_list = get_test_fighter_images_as_pixel_values(jet_types=JET_TYPES)

print("\n************ TEST ************")
test_recognizer(jet_recognizer, test_sample_list, label_encoder, test_set=True)
