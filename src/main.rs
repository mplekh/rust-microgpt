// -----------------------------
// Rust translation of Andrej Karpathy's microgpt
// https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95#file-microgpt-py
// 
// -----------------------------
pub mod mt19937_rng;
pub mod tape;
pub mod model;

use crate::mt19937_rng::PythonRandom;
use crate::tape::*;
use crate::model::*;

const NUM_STEPS: usize = 1000;

fn encode(s: &str, charset: &[char]) -> Vec<usize> {
    s.chars()
        .map(|c| charset.iter().position(|x| *x == c).unwrap())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize Tape and Random Number Generator
    let mut rng = PythonRandom::new(42);
    let mut tape = Tape::new(
        MAX_VOCAB_SIZE * N_EMBD * 3
            + N_LAYER * (4 * N_EMBD * N_EMBD + 4 * N_EMBD * N_EMBD),
    );

    // 2. Load and Shuffle Documents
    let text = std::fs::read_to_string("input.txt").unwrap();

    let mut docs: Vec<&str> =
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .collect();

    rng.shuffle(&mut docs);
    println!("Num docs: {}", docs.len());

    // 3. Build Vocabulary
    let mut charset: Vec<char> = docs.join("").chars().collect();
    charset.sort();
    charset.dedup();
    let bos = charset.len();
    let vocab_size = bos + 1;
    println!("Vocab size: {}", vocab_size);

    if vocab_size > MAX_VOCAB_SIZE {
        panic!("vocab_size ({}) exceeds MAX_VOCAB_SIZE", vocab_size);
    }

    // 4. Initialize Model and Optimizer State
    let model = Model::new(&mut tape, &mut rng, vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER);
    let weights_end = tape.len();
    println!("Num params: {}", weights_end);

    let learning_rate = 0.01;
    let beta1: DataT = 0.85;
    let beta2: DataT = 0.99;
    let eps_adam = 1e-8;

    let mut m = vec![0.0; weights_end];
    let mut v = vec![0.0; weights_end];

    // 5. Training Loop
    for step in 0..NUM_STEPS {
        let doc = &docs[step % docs.len()];
        
        // Tokenizer: BOS + doc characters + BOS
        let mut tokens = vec![bos];
        tokens.extend(encode(doc, &charset));
        if tokens.len() > BLOCK_SIZE {
            tokens.truncate(BLOCK_SIZE);
        }
        tokens.push(bos);

        let loss_idx = model.forward(&mut tape, &tokens);

        // Backward Pass
        tape.backward(loss_idx);

        // Adam Optimizer
        let lr_t = learning_rate * (1.0 - (step as DataT / NUM_STEPS as DataT));
        let beta1_pow = beta1.powi((step + 1) as i32);
        let beta2_pow = beta2.powi((step + 1) as i32);

        for i in 0..weights_end {
            let p_grad = tape.grad[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * p_grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p_grad * p_grad;
            
            let m_hat = m[i] / (1.0 - beta1_pow);
            let v_hat = v[i] / (1.0 - beta2_pow);
            
            tape.data[i] -= lr_t * m_hat / (v_hat.sqrt() + eps_adam);
        }

        if (step + 1) % 10 == 0 {
            print!(
                "Step {} / {} | loss {:.4} | beta1_pow {:.6} | beta2_pow {:.6}\r",
                step + 1, NUM_STEPS, tape.data[loss_idx], beta1_pow, beta1_pow
            );
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        // Reset tape for next step, keeping weights
        tape.truncate(weights_end);
    }

    // 6. Inference
    println!("\n\n-------- inference --------\n");
    let temperature = 0.5;

    for sample_idx in 0..20 {
        let mut keys: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
        let mut values: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
        let mut token_id = bos;
        let mut samples = String::new();

        for pos_id in 0..BLOCK_SIZE {
            let mut logits_indices = [0usize; MAX_VOCAB_SIZE];
            model.gpt(&mut tape, &mut logits_indices, token_id, pos_id, &mut keys, &mut values);
            
            // Apply temperature
            let mut temp_logits = [0usize; MAX_VOCAB_SIZE];
            for i in 0..vocab_size {
                temp_logits[i] = tape.mul_const(logits_indices[i], 1.0 / temperature);
            }

            let mut probs = [0usize; MAX_VOCAB_SIZE];
            tape.softmax(&mut probs, &temp_logits[..vocab_size]);

            // Convert tape indices to actual weight values for choices
            let mut weights = [0.0f32; MAX_VOCAB_SIZE];
            for i in 0..vocab_size {
                weights[i] = tape.data[probs[i]] as f32;
            }

            token_id = rng.choices(&weights[..vocab_size], 1)[0];
            if token_id == bos {
                break;
            }
            samples.push(charset[token_id]);
        }
        
        println!("{}: {}", sample_idx, samples);
        tape.truncate(weights_end);
    }

    Ok(())
}
