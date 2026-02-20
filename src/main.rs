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

fn gpt(
    t: &mut Tape,
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

    let x_emb: Vec<V> = tok_emb.iter().zip(pos_emb)
        .map(|(&tk, &ps)| add(t, tk, ps))
        .collect();

    let mut x = rmsnorm(t, &x_emb);

    // --- Multi-head attention ---
    let x_residual_attn = x.clone();
    let x_norm = rmsnorm(t, &x);

    let q = linear(t, &x_norm, attn_wq);
    let k = linear(t, &x_norm, attn_wk);
    let v = linear(t, &x_norm, attn_wv);

    keys[0].push(k.clone());
    values[0].push(v.clone());

    let mut x_attn: Vec<V> = vec![];
    let scale = 1.0 / (head_dim as f64).sqrt();

    for h in 0..n_head {
        let hs = h * head_dim;
        let q_h = &q[hs..hs + head_dim];
        let mut attn_logits: Vec<V> = vec![];

        for time_step in 0..keys[0].len() {
            let k_h = &keys[0][time_step][hs..hs + head_dim];
            let mut dot = t.val(0.0);
            for j in 0..head_dim {
                let prod = mul(t, q_h[j], k_h[j]);
                dot = add(t, dot, prod);
            }
            attn_logits.push(mul_const(t, dot, scale));
        }

        let attn_weights = softmax(t, &attn_logits);

        for j in 0..head_dim {
            let mut sum = t.val(0.0);
            for time_step in 0..values[0].len() {
                let term = mul(t, attn_weights[time_step], values[0][time_step][hs + j]);
                sum = add(t, sum, term);
            }
            x_attn.push(sum);
        }
    }

    x = linear(t, &x_attn, attn_wo);
    x = x.iter().zip(&x_residual_attn).map(|(&a, &b)| add(t, a, b)).collect();

    // --- MLP block ---
    let x_residual_mlp = x.clone();
    let x_norm_mlp = rmsnorm(t, &x);
    let mut x_mlp = linear(t, &x_norm_mlp, mlp_fc1);
    x_mlp = x_mlp.into_iter().map(|xi| relu(t, xi)).collect();
    x_mlp = linear(t, &x_mlp, mlp_fc2);

    x = x_mlp.iter().zip(&x_residual_mlp).map(|(&a, &b)| add(t, a, b)).collect();

    linear(t, &x, lm_head)
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

    let mut uchars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect();
    uchars.sort_unstable();
    uchars.dedup();

    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    // Hyperparameters
    let n_embd = 16;
    let n_layer = 1;
    let n_head = 4;
    let head_dim = n_embd / n_head;
    let block_size = 16;

    let mut tape = Tape::new();

    // Matrix helper for Tape
    let mut matrix = |t: &mut Tape, nout: usize, nin: usize| -> Vec<Vec<V>> {
        (0..nout).map(|_| (0..nin).map(|_| t.val(rng.gauss(0.0, 0.02))).collect()).collect()
    };

    // Initialize Weight Parameters (These stay on the tape at indices 0..N)
    let wte = matrix(&mut tape, vocab_size, n_embd);
    let wpe = matrix(&mut tape, block_size, n_embd);
    let lm_head = matrix(&mut tape, vocab_size, n_embd);
    let attn_wq = matrix(&mut tape, n_embd, n_embd);
    let attn_wk = matrix(&mut tape, n_embd, n_embd);
    let attn_wv = matrix(&mut tape, n_embd, n_embd);
    let attn_wo = matrix(&mut tape, n_embd, n_embd);
    let mlp_fc1 = matrix(&mut tape, 4 * n_embd, n_embd);
    let mlp_fc2 = matrix(&mut tape, n_embd, 4 * n_embd);

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

    let num_params = params.len();
    println!("num params: {}", num_params);
    // Adam buffers
    let mut m = vec![0.0; num_params];
    let mut v = vec![0.0; num_params];

    // Save the "Watermark" index where weights end and computation begins
    let weights_end_idx = tape.nodes.len();

    let lr = 0.01;
    let beta1 = 0.85;
    let beta2 = 0.99;
    let eps = 1e-8;
    let num_steps = 1000;

    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let mut tokens = vec![bos];
        for ch in doc.chars() {
            tokens.push(uchars.iter().position(|c| *c == ch).unwrap());
        }
        tokens.push(bos);

        let mut keys: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];
        let mut values: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];

        let mut losses = vec![];

        for pos in 0..tokens.len() - 1 {
            let logits = gpt(&mut tape, tokens[pos], pos, &mut keys, &mut values,
                             &wte, &wpe, &lm_head, &attn_wq, &attn_wk, &attn_wv,
                             &attn_wo, &mlp_fc1, &mlp_fc2, n_head, head_dim);
            let probs = softmax(&mut tape, &logits);
            let target_id = tokens[pos + 1];
            let prob_val = probs[target_id];

            let log_val = log(&mut tape, prob_val); // First borrow ends here
            let loss_t = neg(&mut tape, log_val);    // Second borrow starts here

            losses.push(loss_t);
        }

        let mut sum_loss_idx = tape.val(0.0);
        for l_idx in losses {
            sum_loss_idx = add(&mut tape, sum_loss_idx, l_idx);
        }
        let n_val = tape.val((tokens.len() - 1) as f64);
        let total_loss_idx = div(&mut tape, sum_loss_idx, n_val);

        backward(&mut tape, total_loss_idx);
        /*
        let grad_sum: f64 = params
            .iter()
            .map(|&id| tape.nodes[id].grad.abs())
            .sum();

        println!("grad_sum={}", grad_sum);
        */

        // Adam update and Reset
        for (i, &p_idx) in params.iter().enumerate() {
            let g = tape.nodes[p_idx].grad;

            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / (1.0 - beta1.powi((step + 1) as i32));
            let v_hat = v[i] / (1.0 - beta2.powi((step + 1) as i32));

            tape.nodes[p_idx].data -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        let loss_f64 = tape.nodes[total_loss_idx].data;
        print!("step {:4} | loss {:.4}\r", step + 1, loss_f64);

        // Clear the tape of computation nodes, keep weights
        tape.nodes.truncate(weights_end_idx);
        tape.zero_grad(); // Reset weights' grads for next step
    }

    // -----------------------------
    // Inference
    // -----------------------------

    let temperature = 0.5;
    println!("\n--- inference (hallucinated names) ---");

    // Use the watermark from earlier to know where parameters end
    // let weights_end_idx = tape.nodes.len();

    for sample_idx in 0..20 {
        let mut keys: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];
        let mut values: Vec<Vec<Vec<V>>> = vec![vec![]; n_layer];

        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..block_size {
            // 1. Forward pass through GPT
            let logits = gpt(
                &mut tape,
                token_id,
                pos_id,
                &mut keys,
                &mut values,
                &wte, &wpe, &lm_head,
                &attn_wq, &attn_wk, &attn_wv, &attn_wo,
                &mlp_fc1, &mlp_fc2,
                n_head, head_dim,
            );

            // 2. Temperature scaling
            let temp_val = tape.val(temperature);
            let scaled: Vec<V> = logits
                .iter()
                .map(|&l| div(&mut tape, l, temp_val))
                .collect();

            // 3. Softmax to get probabilities
            let probs = softmax(&mut tape, &scaled);

            // 4. Extract data from the tape for sampling
            let weights: Vec<f64> = probs
                .iter()
                .map(|&p| tape.nodes[p].data)
                .collect();

            // 5. Weighted random sampling
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

            // --- TAPE CLEANUP ---
            // We are done with this token's computation graph.
            // Wipe everything after the parameters to keep memory constant.
            tape.nodes.truncate(weights_end_idx);

            token_id = next_token;
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }

}
