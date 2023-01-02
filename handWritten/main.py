import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'mse'])

# model.fit(x_train, y_train, epochs=4)

# val_loss, val_acc, val_mse = model.evaluate(x_test, y_test)

# model.save('handWritten.model')

model = tf.keras.models.load_model('handWritten.model')

loss, acc, mse = model.evaluate(x_test, y_test)

print(loss, acc, mse)

