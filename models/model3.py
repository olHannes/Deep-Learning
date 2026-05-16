import tensorflow as tf
from tensorflow.keras import layers, Sequential

#6,800,522 params
#small but many kernels

def build_model(input_shape=(256, 256, 3), num_classes=10):
    model = Sequential()

    model.add(layers.Conv2D(64, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(256, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(512, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))
    
    model.add(layers.Conv2D(1024, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))


    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model