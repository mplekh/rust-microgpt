# Micro-GPT in Rust
### A high-performance, character-level GPT written in Pure Rust.

This project is a translation of original microgpt.py Python implementation by Andrej Karpathy **[https://gist.github.com/karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)** .
This implementation aligns software logic with hardware reality for maximum speed.

### Features

Core Technical Features

1. **Custom Autograd Engine**:
A high-performance "Tape" based reverse-mode automatic differentiation engine.
It uses a pre-allocated memory arena and unsafe raw pointers to bypass Rust's
bounds-checking overhead in the hot backpropagation loop, matching C++ performance.

1. **Deterministic Python-Style RNG**: A custom implementation of the Mersenne Twister (MT19937)
that replicates Python’s random module exactly.

## Getting Started

### 1. Download Training Data

The model is designed for short-string generation (names, lyrics, or code snippets).
You can use the dataset from the original Gist:

**[Download input.txt here](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)**

Place the downloaded `names.txt` (or any text file) in the project root directory and rename it to `input.txt`.

### 2. Run

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release

```

## How it Works

The model reads `input.txt`, creates a character-level vocabulary, and begins training:

1. **Tape Initialization**: The model allocates a contiguous block of memory for the "Tape"
- a graph of every operation performed during the forward pass.

1. **Forward Pass**:
    Embedding: Combines Token Embeddings (WTE) and Positional Embeddings (WPE).

    Attention: Computes Multi-Head Attention using a scaled dot-product.

    MLP: A two-layer feed-forward network with ReLU activation.

1. **Backpropagation**: The backward function traverses the Tape in reverse,
using raw pointer arithmetic to update gradients efficiently.

1. **Adam Optimizer**: Updates weights using first and second moment estimates (m and v)
with a linear learning rate decay.

After training, the model enters an inference loop and generates 20 new "hallucinated" strings based on the patterns it learned.

## Credits

* Original microgpt.py Python implementation by **Andrej Karpathy**.
