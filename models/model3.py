import tensorflow as tf
from keras import layers, models, Sequential

def build_model(input_shape=(256,256,3), num_clases=10):
    model = Sequential()

    model.add(layers.Conv2D(64, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (5,5), activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(256, (7,7), activation="relu"))
    
    model.add(layers.Conv2D(1024, (7,7), activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    return model