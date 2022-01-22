# From: https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/activations/mish.py#L21-L46
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras


# @tf.function
# def mish(inputs):
#     # return inputs * tf.math.tanh(tf.math.softplus(inputs))
#     # with tf.device('GPU:0'):
#     #     inputs = tf.constant(inputs)
#     # return inputs * tf.math.tanh(tf.math.softplus(inputs))
#     # with tf.device('GPU:0'):
#     #     # inputs_t = tf.convert_to_tensor(inputs)
#     #     inputs_t = tf.constant(inputs)
#     return tf.math.multiply(inputs, tf.math.tanh(tf.math.softplus(inputs)))

# @tf.function(jit_compile=True)  # UNIMPLEMENTED: Could not find compiler for platform METAL
# @tf.function
def mish(inputs):
    # _inputs = tf.cast(inputs, tf.float16)
    # _outputs = _inputs * tf.math.tanh(tf.math.softplus(_inputs))
    # return tf.cast(_outputs, tf.float32)
    return inputs * tf.math.tanh(tf.math.softplus(inputs))
    # inputs = tf.convert_to_tensor(inputs)
    # with tf.device('/GPU:0'):
    #     return inputs * tf.math.tanh(tf.math.softplus(inputs))
    # return tf.math.multiply(inputs, tf.math.tanh(tf.math.softplus(inputs)))
    # sp = tf.math.softplus(inputs)
    # tan_h = tf.math.tanh(sp)
    # return inputs * tan_h
    # return keras.layers.Lambda(lambda inputs: inputs * K.tanh(K.softplus(inputs)))(inputs)
