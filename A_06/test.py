"""Load a trained TwoLayerNet pickle and evaluate it on the test set.

Usage:
    python test.py --dataset iris              # every Iris test sample (10 per class)
    python test.py --dataset mnist             # 5 test samples per class + overall accuracy
    python test.py --dataset mnist --per_class 10
"""
import argparse
import pickle

import numpy as np

import datasets
from train import model_path
import two_layer_net  # noqa: F401  (class definition needed to unpickle the model)


def load_model(dataset):
    with open(model_path(dataset), "rb") as f:
        return pickle.load(f)


def evaluate(dataset, per_class=None, verbose=True):
    net = load_model(dataset)
    x, y = datasets.load(dataset, "test")
    x_in = (x - net.x_mean) / net.x_std if net.x_mean is not None else x
    pred = np.argmax(net.predict(x_in), axis=1)
    names = net.class_names

    # choose which samples to list: all of them, or the first `per_class` of each class
    if per_class is None:
        show = np.arange(len(y))
    else:
        show = np.concatenate([np.where(y == c)[0][:per_class] for c in range(len(names))])

    rows = []
    for i in show:
        rows.append((int(i), names[y[i]], names[pred[i]], "correct" if y[i] == pred[i] else "incorrect"))

    acc = float(np.mean(pred == y))
    conf = np.zeros((len(names), len(names)), dtype=int)
    np.add.at(conf, (y, pred), 1)

    if verbose:
        print(f"Model: {model_path(dataset)}")
        print(f"{'sample':>6} | {'actual':>10} | {'predicted':>10} | result")
        print("-" * 46)
        for idx, a, p, r in rows:
            print(f"{idx:>6} | {a:>10} | {p:>10} | {r}")
        listed_ok = sum(r[3] == "correct" for r in rows)
        print("-" * 46)
        print(f"Listed samples: {listed_ok}/{len(rows)} correct")
        print(f"Accuracy on the full {dataset} test set: {np.sum(pred == y)}/{len(y)} = {acc:.4f}")
        print("Confusion matrix (rows = actual, cols = predicted):")
        print(conf)
    return {"rows": rows, "accuracy": acc, "confusion": conf, "pred": pred, "y": y, "x": x}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iris", "mnist"], required=True)
    ap.add_argument("--per_class", type=int, default=None,
                    help="number of test samples to list per class (default: all for iris, 5 for mnist)")
    args = ap.parse_args()
    per_class = args.per_class if args.per_class is not None else (None if args.dataset == "iris" else 5)
    evaluate(args.dataset, per_class)
