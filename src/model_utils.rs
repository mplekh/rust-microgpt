// -----------------------------
// Model utilities
// -----------------------------

use crate::autograd_value::*;
use std::f64;

pub fn linear(x: &[V], w: &[Vec<V>]) -> Vec<V> {
    w.iter()
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(wi, xi)| mul(wi, xi))
                .reduce(|acc, val| add(&acc, &val))
                .unwrap_or_else(|| val(0.0))
        })
        .collect()
}

pub fn softmax(logits: &[V]) -> Vec<V> {
    let max_val = logits
        .iter()
        .map(|v| v.borrow().data)
        .fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<V> = logits
        .iter()
        .map(|v| exp(&sub(v, &val(max_val))))
        .collect();

    let sum_exp = exps
        .iter()
        .fold(val(0.0), |acc, e| add(&acc, e));

    exps.iter().map(|e| div(e, &sum_exp)).collect()
}

pub fn rmsnorm(x: &[V]) -> Vec<V> {
    let len = x.len() as f64;
    let mut ms = val(0.0);
    for xi in x {
        ms = add(&ms, &mul(xi, xi));
    }
    ms = div(&ms, &val(len));
    let scale = pow(&add(&ms, &val(1e-5)), -0.5);
    x.iter().map(|xi| mul(xi, &scale)).collect()
}

