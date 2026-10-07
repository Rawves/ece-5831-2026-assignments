"""module3.py - Tests the LogicGate class from logic_gate.py."""
import logic_gate as lg

EXPECTED = {
    "AND":  [0, 0, 0, 1],
    "NAND": [1, 1, 1, 0],
    "OR":   [0, 1, 1, 1],
    "NOR":  [1, 0, 0, 0],
    "XOR":  [0, 1, 1, 0],
}
INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def main():
    gate = lg.LogicGate()
    all_ok = True
    for name, expected in EXPECTED.items():
        func = getattr(gate, name.lower() + "_gate")
        print(f"\n{name} gate")
        for (x1, x2), exp in zip(INPUTS, expected):
            func(x1, x2)
            gate.print_output(name)
            if gate.out != exp:
                all_ok = False
                print(f"  FAIL: expected {exp}")
    print("\nAll gates correct!" if all_ok else "\nSome gates FAILED.")


if __name__ == "__main__":
    main()
