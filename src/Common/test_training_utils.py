from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def split_train_validation_data(X, y, test_pct):
    return train_test_split(X, y, test_size=test_pct)


def get_next_training_batch(X, y, iteration, batch_size):
    x_size = len(X)
    index = (iteration * batch_size) % x_size
    end = index + batch_size
    X_data = X[index: end]
    y_data = y[index: end]
    return X_data, y_data


def get_roc_curve(predicted_probabilities, actual):
    fpr, tpr, thresholds = roc_curve(actual, predicted_probabilities)
    return fpr, tpr, thresholds


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def plot_training_validation_loss(training_loss, validation_loss):
    plt.plot(training_loss, color='blue', label='Training Loss')
    plt.plot(validation_loss, color='red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_training_validation_accuracy(training_accuracy, validation_accuracy):
    plt.plot(training_accuracy, color='blue', label='Training Accuracy')
    plt.plot(validation_accuracy, color='red', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
