"""Implementing Callbacks in TensorFlow using the MNIST Dataset"""
import os
import tensorflow as tf
from tensorflow import keras

# Load the data
current_dir = os.getcwd()
data_path = r"C:\Users\Sys\OneDrive\Documents\tensorflow\mnist.npz"
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Normalize pixel values
x_train = x_train / 255.0

# Define your custom callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

# Define the training function
def train_mnist(x_train, y_train):
    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    return history

# Train the model and get the training history
hist = train_mnist(x_train, y_train)

