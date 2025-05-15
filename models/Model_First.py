import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Layer
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy

import tensorflow.keras as keras


class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.attention_dropout = Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention(inputs, inputs)
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)
        res_x = tf.identity(x)
        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)
        x = self.ff_norm(res_x + x)
        return x
    
    
class AttentionModelTrunk(keras.Model):
    def __init__(self, name='AttentionModelTrunk', num_heads=2, head_size=128, ff_dim=None, 
                 num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size 
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        

    def build(self, input_shape):
        if self.ff_dim is None:
            self.ff_dim = self.head_size
        self.attention_layers = [AttentionBlock(num_heads=self.num_heads, head_size=self.head_size, ff_dim=self.ff_dim,
                                                dropout=self.dropout) for _ in range(self.num_layers)]
        super(AttentionModelTrunk, self).build(input_shape)
        
    def call(self, inputs):
        x = inputs
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return tf.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out



def build_full_model(input_shape, model_trunk, n_classes, mlp_units, attention_dropout, mlp_dropout):
    inputs = keras.Input(shape=input_shape)
    
    x = model_trunk(inputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(attention_dropout)(x)
        
    for dim in mlp_units:
        x = layers.Dense( dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)
