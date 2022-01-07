import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from Common import test_training_utils as ttu
from ImageObjectDetectors.CNN4ImagesBase import CNN4ImagesBase
from Common.DL_FilePaths import PROJECT_ROOT

# tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
# physical_devices = tf.config.list_physical_devices()
print(f"physical_devices: {physical_devices}")
# gpu_device = physical_devices[0]
# print(f"gpu_device: {gpu_device}")
# tf.config.set_visible_devices(gpu_device, 'GPU')


class TensorflowDeeperCNN(CNN4ImagesBase):
    def __init__(self,
                 input_shape,
                 n_output,
                 learning_rate=CNN4ImagesBase.DEFAULT_LEARNING_RATE,
                 default_seed=CNN4ImagesBase.DEFAULT_SEED):
        self.n_output = n_output
        self.model = self.construct_model(input_shape, n_output, learning_rate, default_seed)

    def construct_model(self, input_shape, n_output, learning_rate, default_seed):
        print(f'TF Keras backend: {tf.keras.backend}')
        with tf.device('GPU:0'):
            kernel_init = tf.keras.initializers.glorot_uniform(seed=default_seed)

            print(f'Input Shape: {input_shape}')
            layers = self.create_conv_layers(input_shape=input_shape, kernel_init=kernel_init)
            conv_layer_input, conv_layer_2, conv_layer_3, conv_layer_4 = layers

            if n_output == 1:
                output_layer = Dense(units=1, activation='sigmoid', name='output_layer_binary')
                loss = 'binary_crossentropy'
            else:
                output_layer = Dense(units=n_output, activation='softmax', name='output_layer_categorical')
                loss = 'categorical_crossentropy'

            model = Sequential([
                conv_layer_input,
                MaxPool2D(pool_size=(2, 2)),
                conv_layer_2,
                MaxPool2D(pool_size=(2, 2)),
                conv_layer_3,
                MaxPool2D(pool_size=(2, 2)),
                conv_layer_4,
                MaxPool2D(pool_size=(2, 2)),
                Flatten(),
                Dense(units=256, activation='relu'),
                Dropout(0.8),
                Dense(units=128, activation='relu'),
                Dropout(0.6),
                Dense(units=64, activation='relu'),
                Dropout(0.4),
                Dense(units=32, activation='relu'),
                Dropout(0.2),
                output_layer
            ], name='ImageCNN_TF')

            # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
            # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            # model.compile(loss=loss, metrics=['accuracy'])  # Works on Apple Silicon, but model doesn't learn
            # model.compile(optimizer='nadam', loss=loss, metrics=['accuracy'])  # Seems to work better
            # optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            model.summary()
            return model

    def create_conv2d(self, n_filters, kernel_size, stride, kernel_initializer, name, input_shape=None):
        with tf.device('GPU:0'):
            if input_shape is None:
                return Conv2D(filters=n_filters,
                              kernel_size=kernel_size,
                              strides=stride,
                              padding='same',
                              activation='relu',
                              data_format='channels_last',
                              kernel_initializer=kernel_initializer,
                              name=name)
            else:
                return Conv2D(input_shape=input_shape,
                              filters=n_filters,
                              kernel_size=kernel_size,
                              strides=stride,
                              padding='same',
                              activation='relu',
                              data_format='channels_last',
                              kernel_initializer=kernel_initializer,
                              name=name)

    def create_conv_layers(self, input_shape, kernel_init):
        conv_layer_input = self.create_conv2d(input_shape=input_shape,
                                              n_filters=64,
                                              kernel_size=(3, 3),
                                              stride=1,
                                              kernel_initializer=kernel_init,
                                              name='conv64_input_layer')

        conv_layer_2 = self.create_conv2d(n_filters=64,
                                          kernel_size=(4, 4),
                                          stride=1,
                                          kernel_initializer=kernel_init,
                                          name='conv64_k5_layer')

        conv_layer_3 = self.create_conv2d(n_filters=128,
                                          kernel_size=(3, 3),
                                          stride=2,
                                          kernel_initializer=kernel_init,
                                          name='conv128_k3_layer')

        conv_layer_4 = self.create_conv2d(n_filters=256,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          kernel_initializer=kernel_init,
                                          name='conv256_k3_layer')

        return conv_layer_input, conv_layer_2, conv_layer_3, conv_layer_4

    def train(self,
              train_images,
              train_labels,
              batch_size,
              n_epochs,
              train_validation_split=0.2):
        X_train, X_val, y_train, y_val = ttu.split_train_validation_data(train_images,
                                                                         train_labels,
                                                                         train_validation_split)
        with tf.device('GPU:0'):
            history = self.model.fit(x=X_train,
                                     y=y_train,
                                     epochs=n_epochs,
                                     batch_size=batch_size,
                                     validation_data=(X_val, y_val))
            # test_loss, test_accuracy = self.model.evaluate(X_val, y_val)

        # print(f"Test loss: {test_loss:0.4f}, "
        #       f"Test accuracy: {test_accuracy:0.4f}")
        return history

    def train_all(self, train_gen, valid_gen, n_epochs, batch_size):
        with tf.device('GPU:0'):
            history = self.model.fit(train_gen,
                                     epochs=n_epochs,
                                     batch_size=batch_size,
                                     validation_data=valid_gen)
        return history

    def predict_classes(self, input_data):
        # print(f"input data shape: {input_data.shape}")
        pred = self.model.predict(input_data)
        return np.argmax(pred, axis=1)

    def predict(self, input_data, flatten_output=False, one_hot=False):
        predictions = self.predict_classes(input_data)
        if one_hot:
            predictions = tf.one_hot(predictions, depth=self.n_output)
        if flatten_output:
            predictions = predictions.flatten()

        predictions = predictions.numpy()
        return predictions

    def get_full_model_filename(self, model_filename):
        model_filename += '_tf_d'  # Specify using Tensorflow
        return model_filename

    def save_model(self, model_filename, rel_path='models'):
        model_filename = self.get_full_model_filename(model_filename)
        model_filename = self.add_file_type(model_filename)
        self.model.save(f'{PROJECT_ROOT}/{rel_path}/{model_filename}', save_format='h5')

    def load_model(self, model_filename, rel_path='models'):
        model_filename = self.get_full_model_filename(model_filename)
        model_filename = self.add_file_type(model_filename)
        self.model = load_model(f'{PROJECT_ROOT}/{rel_path}/{model_filename}')
