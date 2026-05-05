
import os 
import numpy as np
import cv2 as cv

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


def buildLabels():
    labelDict = {}

    for index, folderName in enumerate(DATASET_SUBFOLDERS):
        oneHot = np.zeros(len(DATASET_SUBFOLDERS), dtype=np.float32)
        oneHot[index] = 1.0
        labelDict[folderName] = oneHot
    return labelDict


def vectorizeImage(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    return img.reshape(-1)


def loadImages():
    labels = buildLabels()

    x = []
    y = []

    counter = 0
    for folder in DATASET_SUBFOLDERS:
        folderPath = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(folderPath):
            print(f"Pfad wurde nicht gefunden: {folderPath}")
            continue

        for file in os.listdir(folderPath):
            if not file.lower().endswith(IMG_FILES):
                continue
            
            counter +=1
            print(f"check image: {counter}")
            filePath = os.path.join(folderPath, file)

            img = cv.imread(filePath)
            if img is None:
                print(f"Fehler beim Laden von {file}")
                continue
                
            img = vectorizeImage(img)
            x.append(img)
            y.append(labels[folder])

loadImages()