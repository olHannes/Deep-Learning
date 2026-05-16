from pathlib import Path
import random
import shutil

# Konfigurationsparameter

SEED = 42

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

SOURCE_DATASET = Path("assets/Plant_leave_diseases_dataset_with_augmentation/tomato_dataset")

TARGET_DATASET = Path("assets/tomato_split_dataset")

random.seed(SEED)

# Klassen

classes = [folder for folder in SOURCE_DATASET.iterdir() if folder.is_dir()]

print(f"Gefundene Klassen: {len(classes)}")

# Split

for class_dir in classes:

    class_name = class_dir.name

    print(f"\nBearbeite Klasse: {class_name}")

    # Alle Bilder sammeln
    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(class_dir.glob(ext))

    images = list(images)

    # Zufällig mischen
    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_SPLIT)
    val_count = int(total_images * VAL_SPLIT)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Zielordner erstellen
    train_target = TARGET_DATASET / "train" / class_name
    val_target = TARGET_DATASET / "val" / class_name
    test_target = TARGET_DATASET / "test" / class_name

    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)
    test_target.mkdir(parents=True, exist_ok=True)

    # Dateien kopieren
    for img in train_images:
        shutil.copy2(img, train_target / img.name)

    for img in val_images:
        shutil.copy2(img, val_target / img.name)

    for img in test_images:
        shutil.copy2(img, test_target / img.name)

    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")

print("\nDataset-Split abgeschlossen.")
print(f"Neues Dataset gespeichert unter:\n{TARGET_DATASET}")