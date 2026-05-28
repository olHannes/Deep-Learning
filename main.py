from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from models.model_manager import get_model
from evaluation.evaluation import evaluate_model

# Thread-Settings
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"

# GPU logic
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

# config parameter
MODEL_CHOICE = "vgg"#5 # choose modelnumber (1-10) from the modelX.py files
FIT = True # True = train, False = load
TEST = True # True = test with model.predict(), False = no testing

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10

BASE_PATH = Path(__file__).resolve().parent

DATASET_PATH = BASE_PATH / "assets" / "tomato_split_dataset"
if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset path not found: ", DATASET_PATH)

SAVE_PATH = BASE_PATH / "models" / "SavedModels" / f"model_{MODEL_CHOICE}.keras"

RESULT_DIR = BASE_PATH / "results"

log_dir = BASE_PATH / "logs" / f"model_{MODEL_CHOICE}"
os.makedirs(log_dir, exist_ok=True)

saved_models_dir = BASE_PATH / "models" / "SavedModels"
os.makedirs(saved_models_dir, exist_ok=True)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def loadDataSplit(dataset_path, img_size, batch_size, seed):

    trainData = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "train",
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    validationData = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "val",
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    testData = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "test",
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = trainData.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    trainData = trainData.prefetch(buffer_size=AUTOTUNE)
    validationData = validationData.prefetch(buffer_size=AUTOTUNE)
    testData = testData.prefetch(buffer_size=AUTOTUNE)

    return trainData, validationData, testData, class_names

trainData, validationData, testData, class_names = loadDataSplit(
    DATASET_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    SEED
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

# call test (evaluation)
if TEST is True:
    results = evaluate_model(
        model = model,
        test_data = testData,
        classes=class_names,
        output_dir=RESULT_DIR,
        model_name=f"model_{MODEL_CHOICE}"
    )