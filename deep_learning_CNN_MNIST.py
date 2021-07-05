import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist


# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training set: 60,000 samples; each sample is a 28x28 grayscale image
# Test set:     10,000 samples; each sample is a 28x28 grayscale image
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# reshape into a 4-D array
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# convert the pixel values from integer to float32 and 
# normalize the pixel values from the range of 0-255 to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


print('X_train range: ', np.min(X_train), ', ', np.max(X_train))
print('X_test range: ', np.min(X_test), ', ', np.max(X_test))

# determine number of pixels in an input image
num_pixels = X_train.shape[1]

# determine number of classes
num_classes = 10

