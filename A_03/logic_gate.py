"""logic_gate.py - Logic gates built from a single perceptron (NumPy).

Each gate stores its weights (w1, w2) and threshold (th) and fires (outputs 1)
when  np.dot(x, w) > th,  otherwise outputs 0.
XOR is not linearly separable, so it is built from a 2-layer network:
XOR(x1, x2) = AND( OR(x1, x2), NAND(x1, x2) ).
"""
import numpy as np


class LogicGate:
    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.th = None
        self.out = None
        self.x1 = self.x2 = None

    def print_output(self, gate):
        """Print the last computed output, e.g. print_output("AND")."""
        if gate in ("AND", "OR", "NAND", "NOR", "XOR"):
            print(f"Output of {gate} logic is: {self.out}, with x1 = {self.x1}, x2 = {self.x2}")
        else:
            print(f"Unknown gate: {gate}")

    def _perceptron(self, x1, x2, w1, w2, th):
        """Store the parameters, then fire if np.dot(x, w) > th."""
        self.w1, self.w2, self.th = w1, w2, th
        self.x1, self.x2 = x1, x2
        x = np.array([x1, x2])
        w = np.array([w1, w2])
        self.out = 1 if np.dot(x, w) > th else 0
        return self.out

    def and_gate(self, x1, x2):
        return self._perceptron(x1, x2, 0.5, 0.5, 0.7)

    def or_gate(self, x1, x2):
        return self._perceptron(x1, x2, 0.5, 0.5, 0.0)

    def nand_gate(self, x1, x2):
        return self._perceptron(x1, x2, -1, -1, -1.5)

    def nor_gate(self, x1, x2):
        return self._perceptron(x1, x2, -1, -1, -0.5)

    def xor_gate(self, x1, x2):
        c = self.or_gate(x1, x2)        # layer 1
        d = self.nand_gate(x1, x2)      # layer 1
        e = self.and_gate(c, d)         # layer 2
        self.x1, self.x2 = x1, x2       # report the original inputs
        self.out = e
        return e


def main():
    gate = LogicGate()
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for name in ("AND", "NAND", "OR", "NOR", "XOR"):
        func = getattr(gate, name.lower() + "_gate")
        for x1, x2 in inputs:
            func(x1, x2)
            gate.print_output(name)


if __name__ == "__main__":
    main()
