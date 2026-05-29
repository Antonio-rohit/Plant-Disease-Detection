import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


PREPROCESSORS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "efficientnetv2": tf.keras.applications.efficientnet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease model.")
    parser.add_argument("--model", default="Model/plant_disease_model.h5")
    parser.add_argument(
        "--data",
        default="Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--preprocessor", choices=PREPROCESSORS.keys(), default="mobilenetv2")
    parser.add_argument("--output", default="reports/evaluation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        label_mode="categorical",
    )
    class_names = dataset.class_names
    preprocess_input = PREPROCESSORS[args.preprocessor]
    dataset = dataset.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(args.model)
    y_proba = model.predict(dataset, verbose=1)
    y_true_one_hot = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_proba, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    try:
        macro_roc_auc = roc_auc_score(y_true_one_hot, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        macro_roc_auc = None

    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_roc_auc_ovr": macro_roc_auc,
        "classes": class_names,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
