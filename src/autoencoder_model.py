from keras.models import Sequential
from keras.layers import Dense

def build_autoencoder(input_shape, latent_dim):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(latent_dim, activation="relu"))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(input_shape[0], activation="sigmoid"))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
