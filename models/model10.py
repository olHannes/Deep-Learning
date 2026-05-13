import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_model(input_shape=(256, 256, 3), num_classes=10):
    model = Sequential()
    
    model.add(layers.Conv2D(16, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(32, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))
    
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(256, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(2,2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    
    return model