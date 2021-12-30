
class MXNetCNN4Images:

    def __init__(self, input_shape, n_output, learning_rate):
        self.model = self.construct_model(input_shape, n_output, learning_rate)

    def construct_model(self,
                        input_shape,
                        n_output,
                        learning_rate):
        pass

    def train(self,
              train_images,
              train_labels,
              batch_size,
              n_epochs,
              train_validation_split=0.2):
        pass

    def predict(self, input_data, flatten_output=False, one_hot=False):
        pass

    def save_model(self, model_filename):
        model_filename += '_mxnet'  # Specify using Torch
        pass

    def load_model(self, model_filename):
        model_filename += '_mxnet'  # Specify using Torch
        pass
