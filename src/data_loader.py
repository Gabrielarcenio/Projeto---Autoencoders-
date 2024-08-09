import numpy as np
from keras.datasets import fashion_mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return (x_train, y_train), (x_test, y_test)

def add_noise(data, noise_factor=0.3):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return noisy_data
