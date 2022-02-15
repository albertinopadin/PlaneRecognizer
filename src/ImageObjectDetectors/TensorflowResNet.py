from warnings import filters
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Activation, Input, Conv2D, MaxPool2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from Common import test_training_utils as ttu
from ImageObjectDetectors.CNN4ImagesBase import CNN4ImagesBase
from Common.DL_FilePaths import PROJECT_ROOT
from tensorflow.keras.utils import get_custom_objects
from Activations.Mish import mish


physical_devices = tf.config.list_physical_devices('GPU')
print(f"physical_devices: {physical_devices}")

KERNEL_SIZE = (3, 3)
STRIDE = 1
# PADDING = 'valid'
PADDING = 'same'  # Has to be same for ResNet blocks
NUM_RESNET_BLOCKS_PER_FILTER_SIZE = 4
MAX_POOL_SIZE = (2, 2)


class TensorflowResNet(CNN4ImagesBase):
    def __init__(self,
                 input_shape,
                 n_output,
                 learning_rate=CNN4ImagesBase.DEFAULT_LEARNING_RATE,
                 activation='relu',
                 dropout=0.5,
                 optimizer='sgd',
                 n_start_filters=32,
                 n_resnet_filter_blocks=2,
                 n_resnet_blocks_per_filter_block=NUM_RESNET_BLOCKS_PER_FILTER_SIZE,
                 default_seed=CNN4ImagesBase.DEFAULT_SEED,
                 filename='tensorflow_deeper'):
        if activation == 'mish':
            # Add mish activation function:
            get_custom_objects().update({'mish': mish})

        # Can't use in mac...
        # Use mixed precision:
        # policy = tf.keras.mixed_precision.Policy('mixed_float16')
        # tf.keras.mixed_precision.set_policy(policy)

        self.n_output = n_output
        self.activation = activation
        self.dropout = dropout
        self._optimizer = optimizer
        self.n_start_filters = n_start_filters
        self.num_resnet_filter_blocks = n_resnet_filter_blocks,
        self.num_resnet_blocks_per_filter_block = n_resnet_blocks_per_filter_block
        model_fn = self.get_full_model_filepath(model_filename=filename)
        self.checkpoint_callback = ModelCheckpoint(
            # filepath=f'{PROJECT_ROOT}/models/checkpoint_{model_fn}',
            filepath=f'{PROJECT_ROOT}/models/{model_fn}',
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        self.model = self.construct_model(input_shape, n_output, learning_rate, activation, dropout, n_resnet_filter_blocks, default_seed)

    def get_output_layer_and_loss(self, n_output):
        if n_output == 1:
            output_layer = Dense(units=1, activation='sigmoid', name='output_layer_binary')
            loss = 'binary_crossentropy'
        else:
            output_layer = Dense(units=n_output, activation='softmax', name='output_layer_categorical')
            loss = 'categorical_crossentropy'
        return output_layer, loss

    def get_optimizer(self, learning_rate):
        if self._optimizer.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif self._optimizer.lower() == 'adam':
            # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif self._optimizer.lower() == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        elif self._optimizer.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        return optimizer

    # Using Keras Functional API since ResNets aren't sequential:
    def construct_model(self, input_shape, n_output, learning_rate, activation, dropout, n_resnet_filter_blocks, default_seed):
        print(f'TF Keras backend: {tf.keras.backend}')
        with tf.device('/GPU:0'):
            kernel_init = tf.keras.initializers.glorot_uniform(seed=default_seed)

            print(f'Input Shape: {input_shape}')

            output_layer, loss = self.get_output_layer_and_loss(n_output)
            optimizer = self.get_optimizer(learning_rate)
            first_block_num_filters = self.n_start_filters * 2

            inputs = Input(shape=input_shape)
            x = Conv2D(filters=self.n_start_filters,
                       kernel_size=KERNEL_SIZE,
                       strides=STRIDE,
                       padding=PADDING,
                       activation=activation,
                       data_format='channels_last',
                       kernel_initializer=kernel_init,
                       name="Conv1")(inputs)
            
            x = Conv2D(filters=first_block_num_filters,
                       kernel_size=KERNEL_SIZE,
                       strides=STRIDE,
                       padding=PADDING,
                       activation=activation,
                       data_format='channels_last',
                       kernel_initializer=kernel_init,
                       name="Conv2")(x)

            x = MaxPool2D(pool_size=MAX_POOL_SIZE)(x)

            block_num_filters = first_block_num_filters
            for filter_block in range(n_resnet_filter_blocks):
                for resnet_block in range(self.num_resnet_blocks_per_filter_block):
                    x = self.resnet_block(x, 
                                          block_num_filters, 
                                          KERNEL_SIZE, 
                                          STRIDE, 
                                          PADDING, 
                                          activation, 
                                          kernel_init, 
                                          f"ResNet_F_{filter_block + 1}_BLK_{resnet_block + 1}")
                if filter_block < n_resnet_filter_blocks - 1:
                    # TODO: This doesn't work - you need intermediate layer to add next resnet block increase in layers
                    # Update number of block filters:
                    block_num_filters *= 2
                    # Intermediate layer to add filters:
                    x = Conv2D(filters=block_num_filters,
                               kernel_size=KERNEL_SIZE,
                               strides=STRIDE,
                               padding=PADDING,
                               activation=activation,
                               data_format='channels_last',
                               kernel_initializer=kernel_init,
                               name=f"ConvUpdateNumFilters_{block_num_filters}")(x)
            
            x = Conv2D(filters=first_block_num_filters,
                       kernel_size=KERNEL_SIZE,
                       strides=STRIDE,
                       padding=PADDING,
                       activation=activation,
                       data_format='channels_last',
                       kernel_initializer=kernel_init,
                       name="ConvLast")(x)

            x = GlobalAveragePooling2D()(x)
            x = Dense(units=256, activation=activation)(x)
            x = Dropout(dropout)(x)
            outputs = output_layer(x)

            model = Model(inputs, outputs)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            model.summary()
            return model

    # With inspiration from:
    # https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
    def resnet_block(self, input_data, n_filters, kernel_size, stride, padding, activation, kernel_initializer, name_prefix):
        x = Conv2D(filters=n_filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    activation=activation,
                    data_format='channels_last',
                    kernel_initializer=kernel_initializer,
                    name=f'{name_prefix}_1_conv2D_{kernel_size[0]}x{kernel_size[1]}_{n_filters}')(input_data)
        x = BatchNormalization()(x)
        x = Conv2D(filters=n_filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    activation=None,  # Activation will be applied after residual addition
                    data_format='channels_last',
                    kernel_initializer=kernel_initializer,
                    name=f'{name_prefix}_2_conv2D_{kernel_size[0]}x{kernel_size[1]}_{n_filters}')(x)
        x = Add()([x, input_data])
        x = Activation(activation)(x)
        return x

    def train(self,
              train_images,
              train_labels,
              batch_size,
              n_epochs,
              train_validation_split=0.2):
        X_train, X_val, y_train, y_val = ttu.split_train_validation_data(train_images,
                                                                         train_labels,
                                                                         train_validation_split)
        with tf.device('/GPU:0'):
            history = self.model.fit(x=X_train,
                                     y=y_train,
                                     epochs=n_epochs,
                                     batch_size=batch_size,
                                     validation_data=(X_val, y_val),
                                     callbacks=[self.checkpoint_callback])
            # test_loss, test_accuracy = self.model.evaluate(X_val, y_val)

        # print(f"Test loss: {test_loss:0.4f}, "
        #       f"Test accuracy: {test_accuracy:0.4f}")
        return history

    def train_all(self, train_gen, valid_gen, n_epochs, batch_size):
        with tf.device('/GPU:0'):
            history = self.model.fit(train_gen,
                                     epochs=n_epochs,
                                     batch_size=batch_size,
                                     validation_data=valid_gen,
                                     callbacks=[self.checkpoint_callback])
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

    def get_full_model_filepath(self, model_filename):
        model_filename += f'_tf_d_{self.activation}_{self.dropout}_do_{self._optimizer}_{self.n_start_filters}_' + \
                          f'sf_bn_{self.num_resnet_filter_blocks}_rfb_{self.num_resnet_blocks_per_filter_block}_rbpfb'
        return model_filename

    def save_model(self, model_filename, rel_path='models', using_checkpoints=True):
        model_filepath = self.get_full_model_filepath(model_filename)
        if using_checkpoints:
            # Load best weights first:
            self.model = load_model(f'{PROJECT_ROOT}/{rel_path}/{model_filepath}')
        model_filename = self.add_file_type(model_filepath)
        self.model.save(f'{PROJECT_ROOT}/{rel_path}/{model_filename}', save_format='h5')

    def load_model(self, model_filename, rel_path='models', is_checkpoint=False):
        model_filepath = self.get_full_model_filepath(model_filename)
        if is_checkpoint:
            self.model = load_model(f'{PROJECT_ROOT}/{rel_path}/{model_filepath}')
        else:
            model_filename = self.add_file_type(model_filepath)
            self.model = load_model(f'{PROJECT_ROOT}/{rel_path}/{model_filename}')
