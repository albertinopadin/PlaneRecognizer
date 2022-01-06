from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from Common.Platforms import in_mac_os
from JetRecognizerPreprocessing import convert_labels_to_one_hot_vectors


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


# From: https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7
def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()