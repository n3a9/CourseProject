from abc import ABC, abstractmethod
import utils
import tensorflow as tf
from keras import layers
from keras.models import Model

# Adapted the model from https://colab.research.google.com/drive/12noBxRkrZnIkHqvmdfFW2TGdOXFtNePM
class CNN_MODEL(tf.keras.Model):
    
    def __init__(self, parameters):
        super(CNN_MODEL, self).__init__(name='CNN-Model')

        # Get the required parameters
        vocabulary_size = parameters['vocabulary_size']
        #max_length = parameters['max_length']
        
        embedding_dimensions = parameters['embedding_dimensions']
        cnn_filters = parameters['cnn_filters']
        dnn_units = parameters['dnn_units']
        dropout_rate = parameters['dropout_rate']
        #kernel_initializer = parameters['kernel_initializer']
        

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)

        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)

        self.last_dense = layers.Dense(units=1,
                                        activation="sigmoid")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)

        layer_1 = self.cnn_layer1(l) 
        layer_1 = self.pool(layer_1) 

        layer_2 = self.cnn_layer2(l) 
        layer_2 = self.pool(layer_2)

        layer_3 = self.cnn_layer3(l)
        layer_3 = self.pool(layer_3) 
        
        concatenated = tf.concat([layer_1, layer_2, layer_3], axis=-1)

        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated)

        model_output = self.last_dense(concatenated)
        
        return model_output