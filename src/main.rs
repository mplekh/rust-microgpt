// -----------------------------
// Rust translation of Andrej Karpathy's microgpt
// https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95#file-microgpt-py
// -----------------------------

mod autograd_value;
mod model_utils;
mod simple_rng;

use crate::autograd_value::*;
use crate::simple_rng::Rng;
use crate::model_utils::*;

use std::f64;
use std::fs;
use std::collections::HashSet;

fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut Vec<Vec<Vec<V>>>,
    values: &mut Vec<Vec<Vec<V>>>,
    wte: &Vec<Vec<V>>,
    wpe: &Vec<Vec<V>>,
    lm_head: &Vec<Vec<V>>,
    attn_wq: &Vec<Vec<V>>,
    attn_wk: &Vec<Vec<V>>,
    attn_wv: &Vec<Vec<V>>,
    attn_wo: &Vec<Vec<V>>,
    mlp_fc1: &Vec<Vec<V>>,
    mlp_fc2: &Vec<Vec<V>>,
    n_head: usize,
    head_dim: usize,
) -> Vec<V> {

    // Token + position embedding
    let tok_emb = &wte[token_id];
    let pos_emb = &wpe[pos_id];

    let mut x: Vec<V> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| add(t, p))
        .collect();

    x = rmsnorm(&x);

    // Only 1 layer implemented (extendable to n_layer loop if needed)
    // -------------------------------------------------------------
    // Multi-head attention
    // -------------------------------------------------------------

    let x_residual = x.clone();

    let x_norm = rmsnorm(&x);

    let q = linear(&x_norm, attn_wq);
    let k = linear(&x_norm, attn_wk);
    let v = linear(&x_norm, attn_wv);

    keys[0].push(k.clone());
    values[0].push(v.clone());

    let mut x_attn: Vec<V> = vec![];

    for h in 0..n_head {
        let hs = h * head_dim;

        let q_h = &q[hs..hs + head_dim];

        let mut attn_logits: Vec<V> = vec![];

        for t in 0..keys[0].len() {
            let k_h = &keys[0][t][hs..hs + head_dim];

            let mut dot = val(0.0);
            for j in 0..head_dim {
                dot = add(&dot, &mul(&q_h[j], &k_h[j]));
            }

            let scaled = div(&dot, &val((head_dim as f64).sqrt()));
            attn_logits.push(scaled);
        }

        let attn_weights = softmax(&attn_logits);

        for j in 0..head_dim {
            let mut sum = val(0.0);
            for t in 0..values[0].len() {
                sum = add(
                    &sum,
                    &mul(&attn_weights[t], &values[0][t][hs + j]),
                );
            }
            x_attn.push(sum);
        }
    }

    let mut x = linear(&x_attn, attn_wo);

    // Residual connection
    x = x.iter()
        .zip(x_residual.iter())
        .map(|(a, b)| add(a, b))
        .collect();

    // -------------------------------------------------------------
    // MLP block
    // -------------------------------------------------------------

    let x_residual = x.clone();

    let x_norm = rmsnorm(&x);

    let mut x = linear(&x_norm, mlp_fc1);
    x = x.iter().map(|xi| relu(xi)).collect();
    x = linear(&x, mlp_fc2);

    x = x.iter()
        .zip(x_residual.iter())
        .map(|(a, b)| add(a, b))
        .collect();

    // Final projection
    let logits = linear(&x, lm_head);

    logits
}

fn main() {
    let mut rng = Rng::new(42);

    let contents = fs::read_to_string("input.txt")
        .expect("failed to read input.txt");

    let mut docs: Vec<String> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();

    println!("num docs: {}", docs.len());

    // Shuffle
    for i in (1..docs.len()).rev() {
        let j = (rng.uniform() * (i as f64 + 1.0)) as usize;
        docs.swap(i, j);
    }

    let mut charset = HashSet::new();
    for d in &docs {
        for ch in d.chars() {
            charset.insert(ch);
        }
    }
    let mut uchars: Vec<char> = charset.into_iter().collect();
    uchars.sort();

    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    // Hyperparameters
    let n_embd = 16;
    let n_layer = 1;
    let n_head = 4;
    let head_dim = n_embd / n_head;
    let block_size = 16;

    fn matrix(rng: &mut Rng, nout: usize, nin: usize) -> Vec<Vec<V>> {
        (0..nout)
            .map(|_| {
                (0..nin)
                    .map(|_| val(rng.gauss(0.0, 0.08)))
                    .collect()
            })
            .collect()
    }

    // Parameters
    let wte = matrix(&mut rng, vocab_size, n_embd);
    let wpe = matrix(&mut rng, block_size, n_embd);
    let lm_head = matrix(&mut rng, vocab_size, n_embd);

    let attn_wq = matrix(&mut rng, n_embd, n_embd);
    let attn_wk = matrix(&mut rng, n_embd, n_embd);
    let attn_wv = matrix(&mut rng, n_embd, n_embd);
    let attn_wo = matrix(&mut rng, n_embd, n_embd);
    let mlp_fc1 = matrix(&mut rng, 4 * n_embd, n_embd);
    let mlp_fc2 = matrix(&mut rng, n_embd, 4 * n_embd);

    let mut params: Vec<V> = vec![];
    for mat in [
        &wte, &wpe, &lm_head,
        &attn_wq, &attn_wk, &attn_wv, &attn_wo,
        &mlp_fc1, &mlp_fc2,
    ] {
        for row in mat {
            for p in row {
                params.push(p.clone());
            }
        }
    }
    println!("num params: {}", params.len());
    // Adam buffers
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];

    let lr = 0.01;
    let beta1 = 0.85;
    let beta2 = 0.99;
    let eps = 1e-8;
    let num_steps = 1_000;

    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let mut tokens = vec![bos];
        for ch in doc.chars() {
            tokens.push(uchars.iter().position(|c| *c == ch).unwrap());
        }
        tokens.push(bos);

        let n = tokens.len() - 1;

        let mut keys: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];
        let mut values: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];

        let mut losses = vec![];

        for pos in 0..n {
            let token_id = tokens[pos];
            let target_id = tokens[pos + 1];

            let logits = gpt(
                token_id,
                pos,
                &mut keys,
                &mut values,
                &wte,
                &wpe,
                &lm_head,
                &attn_wq,
                &attn_wk,
                &attn_wv,
                &attn_wo,
                &mlp_fc1,
                &mlp_fc2,
                n_head,
                head_dim,
            );
            let probs = softmax(&logits);
            let loss_t = neg(&log(&probs[target_id]));
            losses.push(loss_t);
        }

        let mut loss = val(0.0);
        for l in losses {
            loss = add(&loss, &l);
        }
        loss = div(&loss, &val(n as f64));

        backward(&loss);

        // Adam update
        for (i, p) in params.iter().enumerate() {
            let g = p.borrow().grad;
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / (1.0 - beta1.powi((step + 1) as i32));
            let v_hat = v[i] / (1.0 - beta2.powi((step + 1) as i32));

            p.borrow_mut().data -= lr * m_hat / (v_hat.sqrt() + eps);
            p.borrow_mut().grad = 0.0;
        }

        print!("step {:4} | loss {:.4}\r", step + 1, loss.borrow().data);
    }

    // -----------------------------
    // Inference
    // -----------------------------

    let temperature = 0.5;

    println!("\n--- inference (new, hallucinated names) ---");

    for sample_idx in 0..20 {
        let mut keys: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];
        let mut values: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];

        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..block_size {
            let logits = gpt(
                token_id,
                pos_id,
                &mut keys,
                &mut values,
                &wte,
                &wpe,
                &lm_head,
                &attn_wq,
                &attn_wk,
                &attn_wv,
                &attn_wo,
                &mlp_fc1,
                &mlp_fc2,
                n_head,
                head_dim,
            );

            // temperature scaling
            let scaled: Vec<V> = logits
                .iter()
                .map(|l| div(l, &val(temperature)))
                .collect();

            let probs = softmax(&scaled);

            // extract raw probabilities
            let weights: Vec<f64> = probs
                .iter()
                .map(|p| p.borrow().data)
                .collect();

            // weighted random sampling
            let mut cumulative = 0.0;
            let r = rng.uniform();
            let mut next_token = 0;

            let total: f64 = weights.iter().sum();
            for (i, w) in weights.iter().enumerate() {
                cumulative += w / total;
                if r <= cumulative {
                    next_token = i;
                    break;
                }
            }

            token_id = next_token;
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}
