from keras.models import Sequential
from keras.layers import Dense, Activation

def build_classifier(input_size, hidden_units, num_labels):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
