
import os 
import numpy as np
import keras
import tensorflow as tf

DATASET_PATH = r"./Deep-Learning/assets/Plant_leave_diseases_dataset_with_augmentation/"

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

