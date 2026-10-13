import numpy as np


class MultilayerPerceptron:
    """Fully connected network: 3 -> 5 -> 5 -> 2 -> 2.

    Hidden layers 1-3 use sigmoid, the output layer (layer 4) uses ReLU.
    Weight matrices have shape (inputs, outputs), so a layer computes x @ W + b.
    """

    def __init__(self):
        self.params = {
            "W1": np.array([[0.30, -0.10, 0.25, 0.60, -0.45],
                            [0.15, 0.50, -0.35, 0.20, 0.40],
                            [-0.55, 0.25, 0.10, -0.30, 0.35]]),
            "b1": np.array([0.05, -0.05, 0.10, 0.00, -0.10]),

            "W2": np.array([[0.45, -0.25, 0.10, 0.35, -0.15],
                            [-0.30, 0.55, 0.20, -0.40, 0.25],
                            [0.15, 0.05, -0.50, 0.30, 0.45],
                            [0.60, -0.20, 0.35, -0.10, -0.25],
                            [-0.35, 0.40, 0.25, 0.15, -0.05]]),
            "b2": np.array([0.10, 0.05, -0.10, 0.00, 0.15]),

            "W3": np.array([[0.50, -0.60],
                            [-0.25, 0.35],
                            [0.40, 0.20],
                            [-0.45, -0.30],
                            [0.30, 0.55]]),
            "b3": np.array([-0.05, 0.10]),

            "W4": np.array([[0.90, -0.60],
                            [0.40, 1.10]]),
            "b4": np.array([0.05, -0.10]),
        }

    @staticmethod
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    @staticmethod
    def relu(v):
        return np.maximum(0.0, v)

    def forward(self, x):
        p = self.params

        # layer 1 (sigmoid)
        self.h1 = self.sigmoid(x @ p["W1"] + p["b1"])
        # layer 2 (sigmoid)
        self.h2 = self.sigmoid(self.h1 @ p["W2"] + p["b2"])
        # layer 3 (sigmoid)
        self.h3 = self.sigmoid(self.h2 @ p["W3"] + p["b3"])
        # layer 4 / output (relu)
        return self.relu(self.h3 @ p["W4"] + p["b4"])