import tensorflow as tf
from tensorflow.keras import layers, Model

# Station 2 - Transfer Learning
# VGG16 with Training on ImageNet

def build_model(input_shape=(256, 256, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = tf.keras.applications.vgg16.preprocess_input(inputs)

    ##pretrained VGG16 model without head
    base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

    base_model.trainable = False
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model