from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay, 
    accuracy_score, f1_score, 
    precision_score, recall_score
)



def predict_test_data(model, test_data, model_name):
    print(f"make predictions for model '{model_name}'...")
    yTrue = []
    yPred = []

    for images, labels in test_data:
        predictions = model.predict(images, verbose=0)

        pred_classes = np.argmax(predictions, axis=1)
        yTrue.extend(labels.numpy())
        yPred.extend(pred_classes)
        
    return np.array(yTrue), np.array(yPred)


def evaluate_model(model, test_data, classes, output_dir=None, model_name="model"):
    #make predictions
    yTrue, yPred = predict_test_data(model, test_data, model_name)

    print(f"calculate results for model '{model_name}'...")
    #calculate accuracy (use of scikit-learn)
    accuracy = accuracy_score(yTrue, yPred)
    
    #calculate confusion matrix and build matplot diagram
    confMatrix = confusion_matrix(yTrue, yPred)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Confusion Matrix - {model_name}")
    disp = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=60, values_format="d")

    #calculate precision, recall and f1 score of the model
    precision = precision_score(yTrue, yPred, average="macro")
    recall = recall_score(yTrue, yPred, average="macro")
    f1 = f1_score(yTrue, yPred, average="macro")
    

    #check output path and generate folder for results
    if output_dir is not None:
        output_dir = Path(output_dir)
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_dir = Path(".")
    
    #save confusion matrix
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png")
    plt.close(fig)

    report_path = model_dir / "evaluation_report.txt"

    conv_layers = sum(1 for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D))
    dense_layers = sum(1 for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
    class_count = len(classes)

    class_performance = {}

    for i, class_name in enumerate(classes):
        class_mask = (yTrue == i)

        if np.sum(class_mask) > 0:
            class_correct = np.sum(yPred[class_mask] == i)
            class_total = np.sum(class_mask)

            class_accuracy = class_correct / class_total

            class_performance[class_name] = {
                "correct": class_correct,
                "total": class_total,
                "accuracy": class_accuracy
            }

    #write report
    print(f"build report for model '{model_name}'...")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Evaluation Report\n")
        file.write(f"===================================\n\n")

        file.write(f"Model Information\n")
        file.write(f"------------------\n")
        file.write(f"Model name: {model_name}\n")
        file.write(f"Parameter count: {model.count_params()}\n")
        file.write(f"Layer count: {len(model.layers)}\n")
        file.write(f"Conv layer count: {conv_layers}\n")
        file.write(f"Dense layer count: {dense_layers}\n")
        file.write(f"Class count: {class_count}\n\n")

        file.write(f"Metrics\n")
        file.write(f"------------------\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n\n")

        file.write(f"Class Performance\n")
        file.write(f"------------------\n")
        for class_name, values in class_performance.items():
            file.write(
                f"{class_name}:\t"
                f"{values['accuracy']:.2%} "
                f"({values['correct']}/{values['total']})\n"
            )