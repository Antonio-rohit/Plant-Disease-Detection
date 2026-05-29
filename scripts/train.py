import argparse
import json
from pathlib import Path

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam


BACKBONES = {"mobilenetv2": MobileNetV2, "efficientnetv2b0": EfficientNetV2B0, "resnet50": ResNet50}
PREPROCESSORS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "efficientnetv2b0": tf.keras.applications.efficientnet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plant disease image classifier.")
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/valid folders.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model and metadata.")
    parser.add_argument("--backbone", choices=BACKBONES.keys(), default="efficientnetv2b0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fine-tune-at", type=int, default=40, help="Number of final base layers to unfreeze.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def make_dataset(directory: Path, image_size: int, batch_size: int, shuffle: bool, seed: int):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=seed,
    )


def augment_layer():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_model(backbone_name: str, num_classes: int, image_size: int, fine_tune_at: int, learning_rate: float):
    backbone_kwargs = {"include_top": False, "weights": "imagenet", "input_shape": (image_size, image_size, 3)}
    if backbone_name == "efficientnetv2b0":
        backbone_kwargs["include_preprocessing"] = False
    backbone = BACKBONES[backbone_name](**backbone_kwargs)
    backbone.trainable = True
    for layer in backbone.layers[:-fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = augment_layer()(inputs)
    x = PREPROCESSORS[backbone_name](x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(data_dir / "train", args.image_size, args.batch_size, True, args.seed)
    valid_ds = make_dataset(data_dir / "valid", args.image_size, args.batch_size, False, args.seed)
    class_names = train_ds.class_names

    y_train = []
    for _, labels in train_ds.unbatch():
        y_train.append(int(tf.argmax(labels).numpy()))
    class_weights = compute_class_weight("balanced", classes=list(range(len(class_names))), y=y_train)
    class_weight_map = {index: float(weight) for index, weight in enumerate(class_weights)}

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(
        args.backbone,
        num_classes=len(class_names),
        image_size=args.image_size,
        fine_tune_at=args.fine_tune_at,
        learning_rate=args.learning_rate,
    )

    model_path = output_dir / "plant_disease_model.keras"
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        TensorBoard(log_dir=str(output_dir / "tensorboard")),
    ]
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        class_weight=class_weight_map,
        callbacks=callbacks,
    )

    class_indices = {name: index for index, name in enumerate(class_names)}
    (output_dir / "class_indices.json").write_text(json.dumps(class_indices, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
