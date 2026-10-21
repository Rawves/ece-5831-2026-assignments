# ECE 5831 Assignment 05: Forward Propagation (Rock-Paper-Scissors + MNIST)

| File | What it does |
|---|---|
| `rock-paper-scissors.ipynb` | Problem 1: loads my Teachable Machine Keras model, preprocesses a new image, runs forward propagation and classifies it as Rock / Paper / Scissors (plus a play-against-the-computer cell) |
| `mnist.ipynb` | Problem 2: `Mnist` class (784-50-100-10, sigmoid/sigmoid/softmax) with pre-trained weights, checked on the MNIST test set, then tested on 50 images of my own handwriting |
| `tools/prepare_digits.py` | Converts photos of my handwritten digits to 28x28 MNIST-style PNGs |
| `mnist/` | `sample_weight.pkl` + the four MNIST `.gz` files (60k train / 10k test). `mnist.pkl` is created on first run |
| `images/` | my digits, `n_m.png` (digit n, version m), 5 per digit |
| `model/` | `keras_model.h5`, `labels.txt` exported from Teachable Machine |
| `test_images/` | new hand images for the RPS classifier |

## Run
```bash
conda create -n tm_model python=3.10 && conda activate tm_model
pip install tensorflow==2.15.0 pillow numpy matplotlib jupyter
python tools/prepare_digits.py raw_digits images  # raw photos named n_m.jpg -> images/n_m.png
jupyter notebook
```
