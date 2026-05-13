import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Sehr gut
def build_model(input_shape=(256, 256, 3), num_classes=10):
    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(128, (5,5), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(256, (5,5), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(512, (7,7), activation="relu"))
    

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    return model