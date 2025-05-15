import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Layer
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy

import tensorflow.keras as keras


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout,
    transformer_input_size,
    n_classes,
    att_dropout_2=0.01
):
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    for _ in range(num_transformer_blocks):
        input_emdedding = x
        x = layers.MultiHeadAttention( key_dim=head_size, num_heads=num_heads, dropout=dropout )(x, x)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)( x + input_emdedding )
        res = x
    
        # Feed Forward Part 
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=input_emdedding.shape[-1], kernel_size=1)( x )
        x = layers.LayerNormalization(epsilon=1e-6)( x + res )

    x = layers.Flatten( data_format='channels_last' )( x )
    x = layers.Dropout(att_dropout_2)(x)
    for dim in mlp_units:
        x = layers.Dense( dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
