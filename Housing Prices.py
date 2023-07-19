import tensorflow as tf
import numpy as np


# GRADED FUNCTION: house_model
def house_model():
    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    # Compile your model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    return model


# Create an instance of the house model
model = house_model()

# Make a prediction for a new value
new_x = np.array([7.0])
prediction = model.predict(new_x)[0]
print(prediction)


