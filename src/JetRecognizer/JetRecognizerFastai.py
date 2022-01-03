from time import perf_counter
from pathlib import Path
from Common.Platforms import in_mac_os
from Common.DL_FilePaths import SIZE_960_DIR, FIGHTER_JET_SIZE_DIRS, IMG_SIZE, FIGHTER_JET
from ImagePreprocessing.ImagePreprocessing import get_file_paths_in_dir, JPG_EXT
from fastai.vision.all import *
from ImageObjectDetectors.FastaiSimpleModel import FastaiSimpleModel


def get_jet_filepaths_with_labels(size=IMG_SIZE.IS_960, subfolder='train'):
    filepaths = []
    labels = []
    for jet_type in FIGHTER_JET:
        jet_path = FIGHTER_JET_SIZE_DIRS.get(size).get(jet_type)
        train_path = jet_path + f'/{subfolder}'
        fps = get_file_paths_in_dir(train_path)
        for filepath in fps:
            if filepath.endswith(JPG_EXT):
                filepaths.append(filepath)
                labels.append(jet_type.name)

    return filepaths, labels


if in_mac_os():
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_apple_silicon_960'
else:
    JET_RECOGNIZER_MODEL_FILENAME = 'jet_recognizer_A6000_960'

# Set the following flag to load a saved model:
LOAD_EXISTING_MODEL = True if in_mac_os() else False
# Set the following flag to save model after training:
SAVE_MODEL = True

data_path = Path(SIZE_960_DIR)
filepaths, labels = get_jet_filepaths_with_labels()
dls = ImageDataLoaders.from_lists(data_path, filepaths, labels)
jet_recognizer = FastaiSimpleModel(dls)

if LOAD_EXISTING_MODEL:
    jet_recognizer.load_model(JET_RECOGNIZER_MODEL_FILENAME)

EPOCHS = 1
print(f"\n************ Starting training for fastai model, {EPOCHS} epochs in {'macOS' if in_mac_os() else 'Linux'}... ************\n")
_start = perf_counter()
jet_recognizer.fine_tune(epochs=EPOCHS)
# jet_recognizer.fit(epochs=EPOCHS)
_end = perf_counter()
_elapsed = _end - _start
print(
    f"\n************ Training {EPOCHS} epochs took {int(_elapsed / 60)} minutes {int(_elapsed % 60)} seconds). ************\n")

if SAVE_MODEL:
    jet_recognizer.save_model(JET_RECOGNIZER_MODEL_FILENAME)

# print_individual = False
# if print_individual:
#     print("Predictions vs. Ground Truth:")
#     for t_pred, t_label in zip(predictions, validation_labels):
#         print(f'Prediction: {t_pred}, Truth: {t_label}')
#
# print(f"Predictions type: {type(predictions)}")
# print(f"Validation labels type: {type(validation_labels)}")
# print(f"Model accuracy in validation set: {accuracy_score(validation_labels, predictions):0.4f}")
