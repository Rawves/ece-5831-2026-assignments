"""Dataset helpers for Assignment 06 (Iris + MNIST).

dataset/
  iris/iris_train.csv, iris/iris_test.csv     (80/20 stratified split of the 150-sample Iris set)
  mnist/train-images-idx3-ubyte.gz ...         (original MNIST files, 60,000 train / 10,000 test)
"""
import gzip
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
IRIS_DIR = os.path.join(ROOT, "dataset", "iris")
MNIST_DIR = os.path.join(ROOT, "dataset", "mnist")

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]
MNIST_CLASSES = [str(i) for i in range(10)]


def one_hot(labels, num_classes):
    t = np.zeros((labels.size, num_classes))
    t[np.arange(labels.size), labels] = 1
    return t


# ----------------------------------------------------------------- Iris
def make_iris_csv(test_ratio=0.2, seed=42):
    """Create iris_train.csv / iris_test.csv once (stratified split, 10 test samples per class)."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    x_tr, x_te, y_tr, y_te = train_test_split(
        iris.data, iris.target, test_size=test_ratio, stratify=iris.target, random_state=seed)
    os.makedirs(IRIS_DIR, exist_ok=True)
    header = ",".join(IRIS_FEATURES + ["label", "species"])
    for name, x, y in [("iris_train.csv", x_tr, y_tr), ("iris_test.csv", x_te, y_te)]:
        with open(os.path.join(IRIS_DIR, name), "w") as f:
            f.write(header + "\n")
            for row, lab in zip(x, y):
                f.write(",".join(f"{v:.1f}" for v in row) + f",{lab},{IRIS_CLASSES[lab]}\n")


def load_iris_split(split="train"):
    path = os.path.join(IRIS_DIR, f"iris_{split}.csv")
    if not os.path.exists(path):
        make_iris_csv()
    data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4))
    x = data[:, :4].astype(np.float64)
    y = data[:, 4].astype(np.int64)
    return x, y


# ----------------------------------------------------------------- MNIST
_MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def load_mnist_split(split="train"):
    """Returns images flattened to 784 values scaled to [0, 1] and integer labels."""
    img_file, lab_file = _MNIST_FILES[split]
    with gzip.open(os.path.join(MNIST_DIR, img_file), "rb") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(os.path.join(MNIST_DIR, lab_file), "rb") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return x.astype(np.float64) / 255.0, y.astype(np.int64)


def load(dataset, split):
    if dataset == "iris":
        return load_iris_split(split)
    if dataset == "mnist":
        return load_mnist_split(split)
    raise ValueError(f"unknown dataset: {dataset}")
