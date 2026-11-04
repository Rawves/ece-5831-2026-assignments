"""Train a TwoLayerNet with numerical gradients and save it as a pickle file.

Usage:
    python train.py --dataset iris
    python train.py --dataset mnist
"""
import argparse
import json
import pickle
import time

import numpy as np

import datasets
from two_layer_net import TwoLayerNet

LAST_NAME = "arul"

# Default hyper-parameters per dataset
CONFIG = {
    "iris":  dict(hidden_size=10, lr=0.5, iters=1000, batch_size=20, weight_init_std=0.1,
                  log_every=50, val_size=0),
    # Numerical gradient = 2 forward passes per parameter -> keep the hidden layer small for MNIST
    # (784*10 + 10 + 10*10 + 10 = 7,960 parameters  ->  15,920 forward passes per update)
    "mnist": dict(hidden_size=10, lr=1.0, iters=800, batch_size=100, weight_init_std=0.1,
                  log_every=50, val_size=5000),
}


def model_path(dataset):
    return f"{LAST_NAME}_{dataset}_nn_model.pkl"


def train(dataset, seed=7, verbose=True, **overrides):
    cfg = {**CONFIG[dataset], **overrides}
    np.random.seed(seed)
    class_names = datasets.IRIS_CLASSES if dataset == "iris" else datasets.MNIST_CLASSES
    n_classes = len(class_names)

    x, y = datasets.load(dataset, "train")

    # Hold out a validation set (never use the test set during training)
    perm = np.random.permutation(len(x))
    val_idx, tr_idx = perm[:cfg["val_size"]], perm[cfg["val_size"]:]
    x_train, y_train = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # Feature standardisation for Iris (statistics from the training data only)
    x_mean = x_std = None
    if dataset == "iris":
        x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
        x_train = (x_train - x_mean) / x_std

    t_train = datasets.one_hot(y_train, n_classes)
    if len(x_val):
        t_val = datasets.one_hot(y_val, n_classes)

    net = TwoLayerNet(x_train.shape[1], cfg["hidden_size"], n_classes, cfg["weight_init_std"])
    net.class_names, net.x_mean, net.x_std = class_names, x_mean, x_std

    n_params = sum(p.size for p in net.params.values())
    if verbose:
        print(f"[{dataset}] train={len(x_train)} val={len(x_val)} params={n_params} config={cfg}")

    # Subset used to report training accuracy cheaply
    acc_idx = np.random.choice(len(x_train), min(len(x_train), 10000), replace=False)

    history = {"iter": [], "loss": [], "log_iter": [], "train_acc": [], "val_acc": []}
    start = time.time()
    for i in range(1, cfg["iters"] + 1):
        # 1) random mini-batch
        mask = np.random.choice(len(x_train), cfg["batch_size"])
        xb, tb = x_train[mask], t_train[mask]
        # 2) slope (numerical gradient)
        grad = net.numerical_gradient(xb, tb)
        # 3) update weights
        for k in net.params:
            net.params[k] -= cfg["lr"] * grad[k]

        history["iter"].append(i)
        history["loss"].append(float(net.loss(xb, tb)))

        if i % cfg["log_every"] == 0 or i == 1:
            tr_acc = float(net.accuracy(x_train[acc_idx], t_train[acc_idx]))
            va_acc = float(net.accuracy(x_val, t_val)) if len(x_val) else None
            history["log_iter"].append(i)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            if verbose:
                msg = f"iter {i:5d}/{cfg['iters']}  loss={history['loss'][-1]:.4f}  train_acc={tr_acc:.4f}"
                if va_acc is not None:
                    msg += f"  val_acc={va_acc:.4f}"
                print(msg + f"  ({time.time() - start:.0f}s)", flush=True)

    with open(model_path(dataset), "wb") as f:
        pickle.dump(net, f)
    with open(f"{LAST_NAME}_{dataset}_train_history.json", "w") as f:
        json.dump({"config": cfg, **history}, f)
    if verbose:
        print(f"Saved trained network -> {model_path(dataset)}  (training time {time.time() - start:.0f}s)")
    return net, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iris", "mnist"], required=True)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--hidden_size", type=int)
    ap.add_argument("--batch_size", type=int)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "dataset" and v is not None}
    train(args.dataset, **overrides)
