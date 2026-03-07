use crate::Matrix;
use crate::Tape;
use crate::DataT;
use crate::PythonRandom;

pub const MAX_VOCAB_SIZE: usize = 100;
pub const N_LAYER: usize = 1;
pub const BLOCK_SIZE: usize = 16;
pub const N_EMBD: usize = 16;
pub const N_HEAD: usize = 4;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;

// KVCache defined as a nested array for stack-allocation/fixed-size safety
pub type KVCache = [[[usize; N_EMBD]; BLOCK_SIZE]; N_LAYER];

pub struct Layer {
    pub attn_wq: Matrix,
    pub attn_wk: Matrix,
    pub attn_wv: Matrix,
    pub attn_wo: Matrix,
    pub mlp_fc1: Matrix,
    pub mlp_fc2: Matrix,
}

impl Layer {
    pub fn new(n_embd: usize) -> Self {
        Self {
            attn_wq: Matrix::new(n_embd, n_embd),
            attn_wk: Matrix::new(n_embd, n_embd),
            attn_wv: Matrix::new(n_embd, n_embd),
            attn_wo: Matrix::new(n_embd, n_embd),
            mlp_fc1: Matrix::new(4 * n_embd, n_embd),
            mlp_fc2: Matrix::new(n_embd, 4 * n_embd),
        }
    }
}

pub struct Model {
    pub wte: Matrix,
    pub wpe: Matrix,
    pub lm_head: Matrix,
    pub layers: Vec<Layer>,
    pub vocab_size: usize,
}

impl Model {
    pub fn new(
        tape: &mut Tape,
        rng: &mut PythonRandom, // Assuming a similar RNG implementation
        vocab_size: usize,
        n_embd: usize,
        block_size: usize,
        n_layer: usize,
    ) -> Self {
        let mut model = Self {
            wte: Matrix::new(vocab_size, n_embd),
            wpe: Matrix::new(block_size, n_embd),
            lm_head: Matrix::new(vocab_size, n_embd),
            layers: (0..n_layer).map(|_| Layer::new(n_embd)).collect(),
            vocab_size: vocab_size,
        };

        // Helper to initialize matrix weights into the tape
        let mut init_matrix = |m: &mut Matrix| {
            m.data_start = tape.len();
            for _ in 0..(m.rows * m.cols) {
                tape.push_const(rng.gauss(0.0, 0.08) as DataT);
            }
        };

        init_matrix(&mut model.wte);
        init_matrix(&mut model.wpe);
        init_matrix(&mut model.lm_head);
        for layer in &mut model.layers {
            init_matrix(&mut layer.attn_wq);
            init_matrix(&mut layer.attn_wk);
            init_matrix(&mut layer.attn_wv);
            init_matrix(&mut layer.attn_wo);
            init_matrix(&mut layer.mlp_fc1);
            init_matrix(&mut layer.mlp_fc2);
        }

        model
    }

    pub fn forward(&self, tape: &mut Tape, tokens: &[usize]) -> usize {
        let num_tokens = tokens.len() - 1;
        let mut keys: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
        let mut values: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
        let mut losses = [0usize; BLOCK_SIZE];

        for pos_id in 0..num_tokens {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];

            let mut logits = [0usize; MAX_VOCAB_SIZE];
            self.gpt(&mut *tape, &mut logits, token_id, pos_id, &mut keys, &mut values);

            let mut probs = [0usize; MAX_VOCAB_SIZE];
            tape.softmax(&mut probs, &logits[..self.vocab_size]);

            losses[pos_id] = tape.inv_log(probs[target_id]);
        }

        let mut total_losses = losses[0];
        for i in 1..num_tokens {
            total_losses = tape.add(total_losses, losses[i]);
        }
        let loss_idx = tape.mul_const(total_losses, 1.0 / (num_tokens as DataT));
        loss_idx
    }

    pub fn gpt(
        &self,
        tape: &mut Tape,
        logits_out: &mut [usize],
        token_id: usize,
        pos_id: usize,
        keys: &mut KVCache,
        values: &mut KVCache,
    ) {
        let mut x = [0usize; N_EMBD];
        let mut tmp = [0usize; N_EMBD];

        // Joint token and position embedding
        for j in 0..N_EMBD {
            x[j] = tape.add(self.wte.at(token_id, j), self.wpe.at(pos_id, j));
        }

        // RMSNorm and residual preparation
        tape.rmsnorm(&mut tmp, &x);
        x.copy_from_slice(&tmp);

        for i_layer in 0..N_LAYER {
            let mut x_residual = [0usize; N_EMBD];
            x_residual.copy_from_slice(&x);

            tape.rmsnorm(&mut tmp, &x);
            x.copy_from_slice(&tmp);

            // Q, K, V Projections
            let mut q = [0usize; N_EMBD];
            tape.linear(&mut q, &x, &self.layers[i_layer].attn_wq);
            // Note: Writing directly into K/V Cache arrays
            tape.linear(&mut keys[i_layer][pos_id], &x, &self.layers[i_layer].attn_wk);
            tape.linear(&mut values[i_layer][pos_id], &x, &self.layers[i_layer].attn_wv);

            let attn_len = pos_id + 1;
            let mut x_attn = [0usize; N_EMBD];

            // Multi-head Attention
            for h in 0..N_HEAD {
                let hs = h * HEAD_DIM;
                let mut attention_logits = [0usize; BLOCK_SIZE];

                for t in 0..attn_len {
                    let mut sum = tape.mul(q[hs], keys[i_layer][t][hs]);
                    for j in 1..HEAD_DIM {
                        sum = tape.mul_add(q[hs + j], keys[i_layer][t][hs + j], sum);
                    }
                    attention_logits[t] = tape.mul_const(sum, 1.0 / (HEAD_DIM as DataT).sqrt());
                }

                let mut attn_weights = [0usize; BLOCK_SIZE];
                tape.softmax(&mut attn_weights, &attention_logits[..attn_len]);

                // Weighted sum of values
                for j in 0..HEAD_DIM {
                    let mut sum = tape.mul(attn_weights[0], values[i_layer][0][hs + j]);
                    for t in 1..attn_len {
                        sum = tape.mul_add(attn_weights[t], values[i_layer][t][hs + j], sum);
                    }
                    x_attn[hs + j] = sum;
                }
            }

            // Output projection
            tape.linear(&mut x, &x_attn, &self.layers[i_layer].attn_wo);

            // Residual connection
            for i in 0..N_EMBD {
                x[i] = tape.add(x[i], x_residual[i]);
            }

            // MLP Block
            x_residual.copy_from_slice(&x);
            tape.rmsnorm(&mut tmp, &x);
            x.copy_from_slice(&tmp);

            let mut mlp_hidden = [0usize; 4 * N_EMBD];
            tape.linear(&mut mlp_hidden, &x, &self.layers[i_layer].mlp_fc1);
            for i in 0..(4 * N_EMBD) {
                mlp_hidden[i] = tape.relu(mlp_hidden[i]);
            }
            tape.linear(&mut x, &mlp_hidden, &self.layers[i_layer].mlp_fc2);

            for i in 0..N_EMBD {
                x[i] = tape.add(x[i], x_residual[i]);
            }
        }

        tape.linear(logits_out, &x, &self.lm_head);
    }

}
