// -----------------------------
// Model utilities
// -----------------------------

use crate::autograd_value::*;
use std::f64;

pub fn linear(t: &mut Tape, x: &[V], w: &Vec<Vec<V>>) -> Vec<V> {
    w.iter()
        .map(|row| {
            let products: Vec<V> = row.iter()
                .zip(x)
                .map(|(&wi, &xi)| mul(t, wi, xi))
                .collect();

            products.into_iter()
                .reduce(|acc, val| add(t, acc, val))
                .unwrap_or_else(|| t.val(0.0))
        })
        .collect()
}

pub fn softmax(t: &mut Tape, logits: &[V]) -> Vec<V> {
    // 1. Find max value (numerical stability)
    let max_data = logits
        .iter()
        .map(|&v| t.nodes[v].data)
        .fold(f64::NEG_INFINITY, f64::max);

    let max_val = t.val(max_data);

    // 2. Compute exp(x - max)
    let exps: Vec<V> = logits
        .iter()
        .map(|&v| {
            let diff = sub(t, v, max_val);
            exp(t, diff)
        })
        .collect();

    // 3. Sum of exps
    let sum_exp = exps
        .iter()
        .copied()
        .reduce(|acc, e| add(t, acc, e))
        .unwrap_or_else(|| t.val(1e-9)); // Prevent div by zero

    // 4. Normalize
    exps.into_iter().map(|e| div(t, e, sum_exp)).collect()
}

pub fn rmsnorm(t: &mut Tape, x: &[V]) -> Vec<V> {
    let len_val = t.val(x.len() as f64);
    let eps = t.val(1e-5);

    // Mean Square: sum(x^2) / N
    let mut ms = t.val(0.0);
    for &xi in x {
        let sq = mul(t, xi, xi);
        ms = add(t, ms, sq);
    }
    ms = div(t, ms, len_val);

    // Scale: 1 / sqrt(ms + eps)
    let ms_eps = add(t, ms, eps);
    let scale = pow(t, ms_eps, -0.5);

    x.iter().map(|&xi| mul(t, xi, scale)).collect()
}

#[test]
fn test_rmsnorm() {
    let mut t = Tape::new();
    let x = vec![t.val(1.0), t.val(2.0), t.val(3.0)];
    let norm = rmsnorm(&mut t, &x);

    let output_node = norm[2]; // Let's check the last element
    backward(&mut t, output_node);

    assert!(t.nodes[x[2]].grad != 0.0);
    println!("RMSNorm value: {}", t.nodes[output_node].data);
}
