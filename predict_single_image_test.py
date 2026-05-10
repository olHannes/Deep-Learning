from pathlib import Path
import tensorflow as tf
import numpy as np


BASE_PATH = Path(__file__).resolve().parent
CLASS_NAMES = [
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

MODEL_PATH = BASE_PATH / "models" / "SavedModels" / "model_1.keras"

test_image = 50
test_class = 5
IMAGE_PATH = BASE_PATH / "assets" / "Plant_leave_diseases_dataset_with_augmentation" / "tomato_dataset" / f"{CLASS_NAMES[test_class]}" / f"image ({test_image}).jpg"

IMG_SIZE = (256, 256)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU gefunden:", gpus)
else:
    print("Keine GPU gefunden. Vorhersage läuft auf CPU.")


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Modell geladen: {MODEL_PATH}")


if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Bild nicht gefunden: {IMAGE_PATH}")

img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

img_array = tf.keras.utils.img_to_array(img)

img_batch = np.expand_dims(img_array, axis=0)


predictions = model.predict(img_batch)

probabilities = tf.nn.softmax(predictions[0]).numpy()

predicted_index = int(np.argmax(probabilities))
predicted_class = CLASS_NAMES[predicted_index]
confidence = probabilities[predicted_index]

print()
print("Bild:")
print(IMAGE_PATH)

print()
print("Vorhergesagte Klasse:")
print(f"{predicted_class} ({confidence:.2%})")

print()
print("Alle Klassen:")
for class_name, prob in sorted(zip(CLASS_NAMES, probabilities), key=lambda x: x[1], reverse=True):
    print(f"{class_name}: {prob:.2%}")