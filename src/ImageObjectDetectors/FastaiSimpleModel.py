from fastai.vision.all import *
from ImageObjectDetectors.CNN4ImagesBase import CNN4ImagesBase
from Common.DL_FilePaths import PROJECT_ROOT
# How to construct dataloaders for 960 images?


class FastaiSimpleModel(CNN4ImagesBase):
    def __init__(self,
                 input_shape,
                 n_output,
                 learning_rate=CNN4ImagesBase.DEFAULT_LEARNING_RATE,
                 default_seed=CNN4ImagesBase.DEFAULT_SEED):

        self.n_output = n_output
        self.model = self.construct_model(input_shape, n_output, learning_rate, default_seed)

    def construct_model(self, input_shape, n_output, learning_rate, default_seed):
        # return model
        pass

    def train(self,
              train_images,
              train_labels,
              batch_size,
              n_epochs,
              train_validation_split=0.2):
        X_train, X_val, y_train, y_val = ttu.split_train_validation_data(train_images,
                                                                         train_labels,
                                                                         train_validation_split)
        # with tf.device('GPU:0'):
        #     self.model.fit(x=X_train, y=y_train, epochs=n_epochs, batch_size=batch_size)
        #     test_loss, test_accuracy = self.model.evaluate(X_val, y_val)

        print(f"Test loss: {test_loss}, "
              f"Test accuracy: {test_accuracy}")

    def predict_classes(self, input_data):
        pred = self.model.predict(input_data)
        return np.argmax(pred, axis=1)

    def predict(self, input_data, flatten_output=False, one_hot=False):
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
        model_filename = self.add_file_type(model_filename)
        # self.model.save(f'{PROJECT_ROOT}/{rel_path}/{model_filename}', save_format='h5')

    def load_model(self, model_filename, rel_path='models'):
        model_filename = self.get_full_model_filename(model_filename)
        model_filename = self.add_file_type(model_filename)
        # self.model = load_model(f'{PROJECT_ROOT}/{rel_path}/{model_filename}')
