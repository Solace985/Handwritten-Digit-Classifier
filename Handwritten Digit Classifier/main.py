# A Handwritten Digit Classifier program(Image Classifier) using CNN and the MNIST database.
#This program was made as a hands on project for the second week of Season of AI.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255

x_train_final = x_train.reshape(-1, 784)
x_test_final = x_test.reshape(-1, 784)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(784,)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_final, y_train, epochs=10)
model.evaluate(x_test_final, y_test)

predict = model.predict(x_test_final)

output = np.argmax(predict[4])

print(output)