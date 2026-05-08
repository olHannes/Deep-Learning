
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os

from models.model_manager import get_model

MODEL_CHOICE = 1

BASE_PATH = Path(__file__).resolve().parent
DATASET_PATH = BASE_PATH / "assets" / "Plant_leave_diseases_dataset_with_augmentation"
if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset path not found: ", DATASET_PATH)

log_dir = BASE_PATH / "logs"
os.makedirs(log_dir, exist_ok=True)

SAVE_PATH = BASE_PATH / "models" / "SavedModels" / f"model_{MODEL_CHOICE}"

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

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
BATCH_SIZE = 32
SEED = 42


def loadData(dataset_path, dataset_subfolders, type, img_size, batch_size, seed):
    trainData = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        class_names=dataset_subfolders,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset=type,
        seed=seed,
        shuffle=True
    )
    return trainData


trainData = loadData(DATASET_PATH, DATASET_SUBFOLDERS, "training", IMG_SIZE, BATCH_SIZE, SEED)
validationData = loadData(DATASET_PATH, DATASET_SUBFOLDERS, "validation", IMG_SIZE, BATCH_SIZE, SEED)

print("loaded Classes:")
for index, name in enumerate(trainData.class_names):
    print(index, name)


model = get_model(model_choice=MODEL_CHOICE)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    trainData,
    validation_data=validationData,
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=[tensorboard_callback]
)

model.save(SAVE_PATH)
print(f"Model saved at: {SAVE_PATH}")