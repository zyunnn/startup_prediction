# from typing import List, Tuple

from data_loader import DataLoader
from config import Config

import numpy as np
import datetime
import os
import tensorflow as tf
import keras.backend as K
import keras
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda, Conv1D, SpatialDropout1D, Dense, BatchNormalization
from keras.models import Input, Model

from sklearn.metrics import confusion_matrix,classification_report 
from sklearn.metrics import precision_recall_fscore_support


# Define TCN layer
class tcn_layer:
    """
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='linear',
                 kernel_initializer='he_normal',
                 use_batch_norm=False):
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs, training=None):
        x = inputs
        # 1D FCN.
        x = Conv1D(self.nb_filters, 1, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:

                # Residual block for WaveNet TCN
                prev_x = x
                for k in range(2):
                    x = Conv1D(filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=d,
                            kernel_initializer=self.kernel_initializer,
                            padding=self.padding)(x)
                    if self.use_batch_norm:
                        x = BatchNormalization()(x)  
                    x = Activation('relu')(x)
                    x = SpatialDropout1D(rate=self.dropout_rate)(inputs=x, training=training)

                # 1x1 conv to match the shapes (channel dimension)
                prev_x = Conv1D(self.nb_filters, 1, padding='same')(prev_x)
                skip_out = keras.layers.add([prev_x, x])
                skip_out = Activation(self.activation)(skip_out)

                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x


# Metrics for model training
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold

        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater(K.clip(y_true, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.cast(K.greater(K.clip(y_pred,0,1),threshold_value), K.floatx()))
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision


def recall_threshold(threshold=0.5):
    def recall(y_true, y_pred):
        """
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold

        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater(K.clip(y_true, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.cast(K.greater(K.clip(y_true,0,1),threshold_value), K.floatx()))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall


# TCN model
class TCN:

    # Load weight from pre-trained model
    def load_weight(self, filepath):
        self.m.load_weights(filepath)

    # Compile model
    def compile(self):
        batch_size, timesteps, input_dim = None, 6, 70

        i = Input(batch_shape=(batch_size, timesteps, input_dim))
        o = tcn_layer(nb_filters=128,
                      kernel_size=4,
                      dilations=(1,2,4,8,16,32,64),
                      padding='causal',
                      dropout_rate=0.01,
                      return_sequences=False,
                      kernel_initializer='he_normal',
                      use_batch_norm=True)(i)
        o = Dense(1)(o)

        self.m = Model(inputs=[i], outputs=[o])
        self.m.compile(optimizer='adam',loss='hinge', metrics=['accuracy',precision_threshold(0),recall_threshold(0)])


    # Helper function
    # The output should be in the form of 1 and -1 given the hinge loss function
    def reshape(self, y_train):
        for i in range(y_train.shape[0]):
            if y_train[i] == 0:
                y_train[i] = -1
        return y_train

    # Train model
    def train(self, X_train, y_train):
        self.num_epochs = 1 
        y_train = self.reshape(y_train)

        # Save training log
        # current_time = str(datetime.datetime.now())
        # checkpoint_path = "/Users/zhiyun/Desktop/Fall19-20/TEMG4000/startup_prediction/src/model/tcn_training_%s/cp.ckpt"%current_time
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

        # Directory where the checkpoints will be saved
        checkpoint_dir = '/Users/zhiyun/Desktop/Fall19-20/TEMG4000/startup_prediction/src/model'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_1")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_freq=1024)

        self.history = self.m.fit(X_train, y_train, epochs=self.num_epochs, validation_split=0.25, callbacks=[checkpoint_callback])

    # Predict test data
    def predict(self, X_test):
        return self.m.predict(X_test)
    
    # Evaluate model performance with metrics
    def evaluate(self, y_pred, y_test):
        y_pred_rounded = np.clip(y_pred, 0, 1)
        for i in range(y_pred_rounded.shape[0]):
            if y_pred_rounded[i] > 0:
                y_pred_rounded[i] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_rounded)                                                                                                                                                                                                          
        print('Precision:', precision.mean())
        print('Recall:', recall.mean())
        print('f1-score:', f1.mean())
        
        print('Classification report:')
        print(classification_report(y_test, y_pred_rounded))

        print('Confusion matrix:')
        print(confusion_matrix(y_test, y_pred_rounded))

    # Save model
    def save_model(self):
        self.m.save('tcn.hdf5')


if __name__ == "__main__":
    config = Config()
    d = DataLoader(config)
    X_train, y_train = d.get_batch_train()
    X_test, y_test = d.get_batch_test()
    model = TCN()
    model.compile()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(y_pred, y_test)
    # model.save_model()
