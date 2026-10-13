"""Imports and tests the MultilayerPerceptron class using example inputs."""

import numpy as np

from multilayer_perceptron import MultilayerPerceptron


if __name__ == "__main__":
    mlp = MultilayerPerceptron()

    for x in [np.array([1.0, 0.5, -1.0]),
              np.array([0.0, 0.0, 0.0]),
              np.array([-2.0, 1.0, 3.0]),
              np.array([5.0, -3.0, 2.0])]:
        print(f"input: {x}  ->  output: {mlp.forward(x)}")