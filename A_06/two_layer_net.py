"""TwoLayerNet: a 2-layer fully connected network trained with NUMERICAL gradients.

Architecture:  x -> Affine(w1, b1) -> Sigmoid -> Affine(w2, b2) -> Softmax -> y_hat
Loss:          cross-entropy error (averaged over the mini-batch)
Gradient:      central difference  (L(w + h) - L(w - h)) / (2h)  for every parameter
"""
import numpy as np


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["w1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # Extra information saved together with the model (filled in by train.py)
        self.class_names = None   # e.g. ['setosa', 'versicolor', 'virginica']
        self.x_mean = None        # feature normalisation used during training
        self.x_std = None

    # ---------------- activation functions ----------------
    def sigmoid(self, s):
        s = np.clip(s, -500, 500)  # avoid overflow warnings in np.exp
        return 1 / (1 + np.exp(-s))

    def softmax(self, s):
        if s.ndim == 1:  # single sample (batch size = 1)
            s = s - np.max(s)
            exp_s = np.exp(s)
            return exp_s / np.sum(exp_s)
        s = s - np.max(s, axis=1, keepdims=True)
        exp_s = np.exp(s)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    # ---------------- forward pass ----------------
    def predict(self, x):
        w1, w2 = self.params["w1"], self.params["w2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = x @ w1 + b1
        z1 = self.sigmoid(a1)
        a2 = z1 @ w2 + b2
        y_hat = self.softmax(a2)
        return y_hat

    # ---------------- loss ----------------
    def cross_entropy_error(self, y, y_hat):
        if y.ndim == 1:
            y = y.reshape((1, y.size))
            y_hat = y_hat.reshape((1, y_hat.size))
        batch_size = y.shape[0]
        return -np.sum(y * np.log(y_hat + 1e-7)) / batch_size

    def loss(self, x, y):
        y_hat = self.predict(x)
        return self.cross_entropy_error(y, y_hat)

    def accuracy(self, x, y):
        """y can be one-hot or integer labels."""
        pred = np.argmax(self.predict(x), axis=1)
        true = np.argmax(y, axis=1) if y.ndim != 1 else y
        return np.mean(pred == true)

    # ---------------- numerical gradient ----------------
    def numerical_gradient(self, x, y):
        h = 1e-4
        grad = {}
        for k in self.params.keys():
            grad[k] = np.zeros_like(self.params[k])
            # visit every element of the parameter array (any number of dimensions)
            it = np.nditer(self.params[k], flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                tmp = self.params[k][idx]

                self.params[k][idx] = tmp + h
                fh1 = self.loss(x, y)          # f(w + h)

                self.params[k][idx] = tmp - h
                fh2 = self.loss(x, y)          # f(w - h)

                grad[k][idx] = (fh1 - fh2) / (2 * h)
                self.params[k][idx] = tmp      # restore
                it.iternext()
        return grad
