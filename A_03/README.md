# A_03 – Logic Gates with a Perceptron

ECE 5831: Pattern Recognition and Neural Networks (Fall 2026)

This assignment implements the logic gates AND, NAND, OR, NOR and XOR using a perceptron, with NumPy for the computation.

## Files

| File | Description |
|------|-------------|
| `logic_gate.py` | `LogicGate` class with `and_gate`, `nand_gate`, `or_gate`, `nor_gate` and `xor_gate`. Running it directly prints every truth table. |
| `module3.py` | Test script that checks each gate against its expected truth table and prints each output and flags any mismatch. |
| `module3.ipynb` | Documented notebook (headings and explanations) that demonstrates and verifies each gate. |

## How it works

Each gate is a two-input perceptron that stores its weights and threshold on the object and fires when:

```
np.dot([x1, x2], [w1, w2]) > th
```

| Gate | Weights (w1, w2) | Threshold th |
|------|------------------|--------------|
| AND  | 0.5, 0.5 | 0.7  |
| OR   | 0.5, 0.5 | 0.0  |
| NAND | -1, -1   | -1.5 |
| NOR  | -1, -1   | -0.5 |

**XOR** cannot be solved by a single perceptron because it is not linearly separable. It is built as a two-layer network:

```
XOR(x1, x2) = AND( OR(x1, x2), NAND(x1, x2) )
```

After calling a gate, `print_output("AND")` (or OR, NAND, NOR, XOR) prints the stored output and inputs.

## Usage

Requires Python 3 and NumPy (`pip install numpy`).

```bash
python logic_gate.py   # print truth tables for all gates
python module3.py      # run the tests
```

To view the notebook, open `module3.ipynb` in VS Code (with the Python and Jupyter extensions) and choose **Run All**.

## Expected output (truth tables)

| x1 | x2 | AND | NAND | OR | NOR | XOR |
|----|----|-----|------|----|-----|-----|
| 0  | 0  | 0   | 1    | 0  | 1   | 0   |
| 0  | 1  | 0   | 1    | 1  | 0   | 1   |
| 1  | 0  | 0   | 1    | 1  | 0   | 1   |
| 1  | 1  | 1   | 0    | 1  | 0   | 0   |
