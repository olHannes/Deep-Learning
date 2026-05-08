import os 
import numpy as np
import keras
import tensorflow as tf
from keras import layers, models

DATASET_PATH = r"./assets/Plant_leave_diseases_dataset_with_augmentation/"

DATASET_SUBFOLDERS = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

IMG_SIZE = (256, 256)
IMG_FILES = ".jpg"

def build_tomato_model(input_shape=(256, 256, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="tomato_classifier")

model = build_tomato_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="categorical",
    class_names=DATASET_SUBFOLDERS,
    image_size=IMG_SIZE,
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="categorical",
    class_names=DATASET_SUBFOLDERS,
    image_size=IMG_SIZE,
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

loss, acc = model.evaluate(val_ds)
print("Validation loss:", loss)
print("Validation accuracy:", acc)

# Example usage
img = tf.keras.preprocessing.image.load_img("path/to/your/image.jpg", target_size=IMG_SIZE)
img = tf.keras.preprocessing.image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print("Predicted class:", np.argmax(pred))