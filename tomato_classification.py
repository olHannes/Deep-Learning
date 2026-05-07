
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_PATH = Path(__file__).resolve().parent
DATASET_PATH = BASE_PATH / "assets" / "Plant_leave_diseases_dataset_with_augmentation"
if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset path not found: ", DATASET_PATH)

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




def loadData(type):
    trainData = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        labels="inferred",
        label_mode="int",
        class_names=DATASET_SUBFOLDERS,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset=type,
        seed=SEED,
        shuffle=True
    )
    return trainData


trainData = loadData("training")
validationData = loadData("validation")

print("loaded Classes:")
for index, name in enumerate(trainData.class_names):
    print(index, name)

