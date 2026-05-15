from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from models.model_manager import get_model
from evaluation.evaluation import evaluate_model

# Thread-Einstellungen möglichst vor TensorFlow setzen
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"

# GPU Logik
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Verwendete GPUs:", gpus)
    except RuntimeError as e:
        print("GPU-Konfiguration konnte nicht geändert werden:", e)
else:
    print("Keine GPU gefunden. Training läuft auf CPU.")

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Konfigurationsparameter
MODEL_CHOICE = 10  # Wählen der Modellnummer (1-10) aus den modelX.py Dateien
FIT = True # True = Trainieren, False = Laden
TEST = False # True = Test mit model.predict(), False = Kein Test

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
SEED = 42
EPOCHS = 10

BASE_PATH = Path(__file__).resolve().parent

DATASET_PATH = BASE_PATH / "assets" / "Plant_leave_diseases_dataset_with_augmentation" / "tomato_dataset"
if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset path not found: ", DATASET_PATH)

SAVE_PATH = BASE_PATH / "models" / "SavedModels" / f"model_{MODEL_CHOICE}.keras"

RESULT_DIR = BASE_PATH / "results"

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

log_dir = BASE_PATH / "logs"
os.makedirs(log_dir, exist_ok=True)

saved_models_dir = BASE_PATH / "models" / "SavedModels"
os.makedirs(saved_models_dir, exist_ok=True)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def loadDataSplit(dataset_path, dataset_subfolders, img_size, batch_size, seed):
    # Alle Daten laden
    all_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        class_names=dataset_subfolders,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True
    )
    
    # class_names vor dem Split speichern
    class_names = all_data.class_names
    
    # Split: 60 Train, 20 Val, 20 Test
    total_batches = tf.data.experimental.cardinality(all_data).numpy()
    train_size = int(0.6 * total_batches)
    val_size = int(0.2 * total_batches)
    
    trainData = all_data.take(train_size)
    remaining = all_data.skip(train_size)
    
    validationData = remaining.take(val_size)
    testData = remaining.skip(val_size)
    
    return trainData, validationData, testData, class_names

trainData, validationData, testData, class_names = loadDataSplit(
    DATASET_PATH, DATASET_SUBFOLDERS, IMG_SIZE, BATCH_SIZE, SEED
)

print("Loaded classes:")
for index, name in enumerate(class_names):
    print(index, name)

model = get_model(model_choice=MODEL_CHOICE, input_shape=IMG_SIZE + (3,), num_classes=len(class_names))

if FIT is True:
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        trainData,
        validation_data=validationData,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback]
    )

    model.save(SAVE_PATH)
    print(f"Model saved at: {SAVE_PATH}")

else:
    if not SAVE_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SAVE_PATH}")
    model = tf.keras.models.load_model(SAVE_PATH)
    print(f"Model {MODEL_CHOICE} loaded from: {SAVE_PATH}")

# Test mit model.predict()
def evaluate_on_test_data(model, testData, class_names):
    print(f"Testing Model {MODEL_CHOICE} on Test Data...")
    
    all_predictions = []
    all_true_labels = []
    
    for images, labels in testData:
        predictions = model.predict(images, verbose=0)
        probabilities = tf.nn.softmax(predictions).numpy()
        
        pred_classes = np.argmax(probabilities, axis=1)
        all_predictions.extend(pred_classes)
        all_true_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Accuracy berechnen
    correct_predictions = np.sum(all_predictions == all_true_labels)
    total_predictions = len(all_predictions)
    test_accuracy = correct_predictions / total_predictions
    
    print(f"Test Accuracy: {test_accuracy:.4%} ({correct_predictions}/{total_predictions})")
    
    # Einfaches Classification Report für jede Klasse
    print("\nKlassen-Performance:")
    for i, class_name in enumerate(class_names):
        class_mask = (all_true_labels == i)
        if np.sum(class_mask) > 0:
            class_correct = np.sum(all_predictions[class_mask] == i)
            class_total = np.sum(class_mask)
            class_acc = class_correct / class_total
            print(f"  {class_name}: {class_acc:.2%} ({class_correct}/{class_total})")
    
    return test_accuracy, all_predictions, all_true_labels

# Test aufrufen
if TEST is True:
    results = evaluate_model(
        model = model,
        test_data = testData,
        classes=class_names,
        output_dir=RESULT_DIR,
        model_name=f"model_{MODEL_CHOICE}"
    )