"""TensorFlow/Keras product-category classification MVP.

Two data modes:
* fashion: a reproducible Fashion-MNIST demonstration.
* folder: real, already separated train/val/test product photographs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


FASHION_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
SEED = 42


def limited_indices(labels: np.ndarray, maximum: int | None) -> np.ndarray:
    indices = np.arange(len(labels))
    if maximum is None or maximum >= len(indices):
        return indices
    if maximum < len(np.unique(labels)):
        raise ValueError("The sample limit must include at least one example per class.")
    selected, _ = train_test_split(
        indices, train_size=maximum, stratify=labels, random_state=SEED
    )
    return selected


def make_fashion_data(args: argparse.Namespace):
    (images, labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_ids, val_ids = train_test_split(
        np.arange(len(labels)), test_size=0.15, stratify=labels, random_state=SEED
    )
    train_ids = train_ids[limited_indices(labels[train_ids], args.max_train)]
    val_ids = val_ids[limited_indices(labels[val_ids], args.max_val)]
    test_ids = limited_indices(test_labels, args.max_test)

    def dataset(x, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x[..., None], y))
        if shuffle:
            ds = ds.shuffle(len(y), seed=SEED, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train = dataset(images[train_ids], labels[train_ids], True)
    val = dataset(images[val_ids], labels[val_ids])
    test = dataset(test_images[test_ids], test_labels[test_ids])
    counts = dict(train=len(train_ids), val=len(val_ids), test=len(test_ids))
    return train, val, test, test_labels[test_ids], FASHION_NAMES, counts


def make_folder_data(args: argparse.Namespace):
    root = Path(args.data_dir)
    if not root.is_dir():
        raise ValueError(f"Missing data directory: {root}")
    expected = None
    datasets = []
    counts = {}
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f"Missing split directory: {folder}")
        names = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if len(names) < 2:
            raise ValueError(f"{folder}: expected at least two class directories")
        if expected is None:
            expected = names
        elif names != expected:
            raise ValueError(f"{folder}: class names differ from train: {expected}")
        ds = tf.keras.utils.image_dataset_from_directory(
            folder, labels="inferred", label_mode="int", class_names=expected,
            image_size=(args.image_size, args.image_size), batch_size=args.batch_size,
            shuffle=(split == "train"), seed=SEED,
        )
        counts[split] = sum(int(len(batch_labels)) for _, batch_labels in ds)
        datasets.append(ds.prefetch(tf.data.AUTOTUNE))
    test_labels = np.concatenate([y.numpy() for _, y in datasets[2]])
    return *datasets, test_labels, expected, counts


def fashion_model() -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(FASHION_NAMES), activation="softmax"),
    ])


def photo_model(image_size: int, n_classes: int, backbone: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.RandomRotation(0.04)(inputs)
    x = tf.keras.layers.RandomZoom(0.08)(x)
    if backbone == "mobilenet":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), include_top=False,
            weights="imagenet"
        )
        base.trainable = False
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = tf.keras.layers.Rescaling(1.0 / 255)(x)
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "fashion":
        train_ds, val_ds, test_ds, y_true, classes, counts = make_fashion_data(args)
        model = fashion_model()
        image_size = 28
    else:
        train_ds, val_ds, test_ds, y_true, classes, counts = make_folder_data(args)
        image_size = args.image_size
        model = photo_model(image_size, len(classes), args.backbone)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    best_path = out / "model.keras"
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, shuffle=False, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(out / "history.csv"),
        ],
    )
    model = tf.keras.models.load_model(best_path)
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=classes, output_dict=True, zero_division=0,
        ),
        "counts": counts,
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "dataset": args.dataset,
    }
    metadata = {
        "classes": classes, "dataset": args.dataset,
        "image_size": image_size,
        "input_mode": "L" if args.dataset == "fashion" else "RGB",
        "backbone": args.backbone if args.dataset == "folder" else "small_cnn",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *classes])
        for label, row in zip(classes, confusion_matrix(y_true, y_pred, labels=range(len(classes)))):
            writer.writerow([label, *row])
    print(f"Saved model and test results to {out}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}; macro F1: {metrics['test_macro_f1']:.4f}")


def image_to_array(source: Image.Image, metadata: dict) -> np.ndarray:
    size = metadata["image_size"]
    image = source.convert(metadata["input_mode"]).resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32)
    if metadata["input_mode"] == "L":
        values = values[..., None]
    return values[None, ...]


def load_image(path: Path, metadata: dict) -> np.ndarray:
    with Image.open(path) as source:
        return image_to_array(source, metadata)


def predict(args: argparse.Namespace) -> None:
    out = Path(args.output)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(out / "model.keras")
    image = load_image(Path(args.image), metadata)
    scores = model.predict(image, verbose=0)[0]
    ranked = np.argsort(scores)[::-1][:3]
    for rank in ranked:
        print(f"{metadata['classes'][rank]}: {scores[rank]:.1%}")
    if metadata["dataset"] == "fashion":
        print("Demo model: expects Fashion-MNIST-style 28x28 grayscale images, not product photographs.")


def sample(args: argparse.Namespace) -> None:
    (_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    index = args.index
    if not 0 <= index < len(test_images):
        raise ValueError(f"index must be between 0 and {len(test_images) - 1}")
    path = Path(args.image)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(test_images[index], mode="L").save(path)
    print(f"Saved {path}; true class: {FASHION_NAMES[test_labels[index]]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p_train = commands.add_parser("train", help="Train and evaluate on a separate test split")
    p_train.add_argument("--dataset", choices=["fashion", "folder"], default="fashion")
    p_train.add_argument("--data-dir", help="Folder containing train/val/test subdirectories")
    p_train.add_argument("--output", default="artifacts")
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--learning-rate", type=float, default=0.001)
    p_train.add_argument("--image-size", type=int, default=160)
    p_train.add_argument("--backbone", choices=["cnn", "mobilenet"], default="cnn")
    p_train.add_argument("--max-train", type=int, default=None)
    p_train.add_argument("--max-val", type=int, default=None)
    p_train.add_argument("--max-test", type=int, default=None)
    p_train.set_defaults(func=train)
    p_predict = commands.add_parser("predict", help="Classify one image")
    p_predict.add_argument("--image", required=True)
    p_predict.add_argument("--output", default="artifacts")
    p_predict.set_defaults(func=predict)
    p_sample = commands.add_parser("sample", help="Export a held-out Fashion-MNIST test image")
    p_sample.add_argument("--index", type=int, default=0)
    p_sample.add_argument("--image", default="sample.png")
    p_sample.set_defaults(func=sample)
    args = parser.parse_args()
    if args.command == "train" and args.dataset == "folder" and not args.data_dir:
        parser.error("folder mode requires --data-dir")
    args.func(args)


if __name__ == "__main__":
    main()
