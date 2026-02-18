# Micro-GPT in Rust

This is a character-level language model written in Rust.

This project is a translation of **[https://gist.github.com/karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)** .

### Features

* **Pure Rust Implementation**: No dependencies on heavy ML frameworks.
* **Autograd Engine**: Manual backpropagation through a dynamic graph.
* **Modern Transformer Architecture**: Includes RMSNorm, Multi-Head Attention, and an MLP block.
* **Inference with KV Caching**: Optimized sampling by passing mutable Key/Value buffers.
* **Adam Optimizer**: A hand-rolled Adam update loop for parameter training.

## Getting Started

### 1. Download Training Data

The model is designed to be trained on a list of names or short strings. You can use the dataset from the original Gist:

**[Download input.txt here](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)**

Place the downloaded `names.txt` (or any text file) in the project root directory and rename it to `input.txt`.

### 2. Run

```bash
cargo run --release

```

## How it Works

The model reads `input.txt`, creates a character-level vocabulary, and begins training:

1. **Embedding**: Maps characters to vectors and adds positional encoding.
2. **Transformer Block**: Processes the sequence through Multi-Head Attention and a Feed-Forward MLP.
3. **Loss**: Calculates cross-entropy loss against the next character in the sequence.
4. **Backprop**: The Autograd engine computes gradients for all parameters.
5. **Optimization**: Adam updates the weights to minimize the loss.

After training, the model enters an inference loop and generates 20 new "hallucinated" strings based on the patterns it learned.

## Credits

* Original Python implementation by **Andrej Karpathy**.
