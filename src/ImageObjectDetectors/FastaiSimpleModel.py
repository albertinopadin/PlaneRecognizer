from fastai.vision.all import *
from Common.DL_FilePaths import PROJECT_ROOT


class FastaiSimpleModel:
    def __init__(self, data_loaders):
        self.model = cnn_learner(data_loaders, resnet18, metrics=accuracy)
        # self.model = cnn_learner(data_loaders, resnet18, lr=learning_rate, metrics=accuracy)

    def fine_tune(self, epochs=4):
        self.model.fine_tune(epochs)

    def fit(self, epochs=4):
        self.model.fit(epochs)

    # def train(self,
    #           train_images,
    #           train_labels,
    #           batch_size,
    #           n_epochs,
    #           train_validation_split=0.2):
    #     X_train, X_val, y_train, y_val = ttu.split_train_validation_data(train_images,
    #                                                                      train_labels,
    #                                                                      train_validation_split)
    #     # with tf.device('GPU:0'):
    #     #     self.model.fit(x=X_train, y=y_train, epochs=n_epochs, batch_size=batch_size)
    #     #     test_loss, test_accuracy = self.model.evaluate(X_val, y_val)
    #
    #     print(f"Test loss: {test_loss}, "
    #           f"Test accuracy: {test_accuracy}")

    # def predict_classes(self, input_data):
    #     pred = self.model.predict(input_data)
    #     return np.argmax(pred, axis=1)

    def predict(self, item):
        return self.model.predict(item)

    # def predict(self, input_data, flatten_output=False, one_hot=False):
        # predictions = self.predict_classes(input_data)
        # if one_hot:
        #     predictions = tf.one_hot(predictions, depth=self.n_output)
        # if flatten_output:
        #     predictions = predictions.flatten()
        #
        # predictions = predictions.numpy()
        # return predictions

    def get_full_model_filename(self, model_filename):
        model_filename += '_fastai'  # Specify using fastai
        return model_filename

    def save_model(self, model_filename, rel_path='models'):
        model_filename = self.get_full_model_filename(model_filename)
        self.model.save(f'{PROJECT_ROOT}/{rel_path}/{model_filename}', with_opt=True)

    def load_model(self, model_filename, rel_path='models'):
        model_filename = self.get_full_model_filename(model_filename)
        self.model.load(f'{PROJECT_ROOT}/{rel_path}/{model_filename}', with_opt=True)
