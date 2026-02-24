// -----------------------------
// Autograd Value
// "Micrograd-style" implementation of reverse-mode automatic differentiation
// -----------------------------

pub type V = usize;

pub struct Tape {
    pub data: Vec<f64>,
    pub grad: Vec<f64>,
    pub children: Vec<[V; 2]>,
    pub local_grads: Vec<[f64; 2]>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            grad: Vec::new(),
            children: Vec::new(),
            local_grads: Vec::new(),
        }
    }

    #[inline(always)]
    fn push(
        &mut self,
        data: f64,
        children: [V; 2],
        local_grads: [f64; 2],
    ) -> V {
        let id = self.data.len();

        self.data.push(data);
        self.grad.push(0.0);
        self.children.push(children);
        self.local_grads.push(local_grads);

        id
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.data.truncate(new_len);
        self.grad.truncate(new_len);
        self.children.truncate(new_len);
        self.local_grads.truncate(new_len);
    }

    pub fn val(&mut self, data: f64) -> V {
        self.push(data, [usize::MAX, usize::MAX], [0.0, 0.0])
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    #[inline(always)]
    pub fn node_count(&self) -> usize {
        self.data.len()
    }
}


pub fn add(t: &mut Tape, a: V, b: V) -> V {
    let data = t.data[a] + t.data[b];
    t.push(data, [a, b], [1.0, 1.0])
}

pub fn mul(t: &mut Tape, a: V, b: V) -> V {
    let data = t.data[a] * t.data[b];
    t.push(data, [a, b], [t.data[b], t.data[a]])
}

pub fn neg(t: &mut Tape, a: V) -> V {
    let data = -t.data[a];
    t.push(data, [a, usize::MAX], [-1.0, 0.0])
}

pub fn sub(t: &mut Tape, a: V, b: V) -> V {
    let data = t.data[a] - t.data[b];
    t.push(data, [a, b], [1.0, -1.0])
}

pub fn div(t: &mut Tape, a: V, b: V) -> V {
    let data = t.data[a] / t.data[b];
    t.push(data, [a, b], [1.0 / t.data[b], - t.data[a] / (t.data[b] * t.data[b])])
}

pub fn mul_const(t: &mut Tape, a: V, c: f64) -> V {
    let data = t.data[a] * c;
    t.push(data, [a, usize::MAX], [c, 0.0])
}

pub fn pow(t: &mut Tape, a: V, p: f64) -> V {
    let a_data = t.data[a];
    let data = a_data.powf(p);
    t.push(data, [a, usize::MAX], [p * a_data.powf(p - 1.0), 0.0])
}

pub fn exp(t: &mut Tape, a: V) -> V {
    let a_data = t.data[a];
    let e = a_data.exp();
    t.push(e, [a, usize::MAX], [e, 0.0])
}

pub fn log(t: &mut Tape, a: V) -> V {
    let a_data = t.data[a];
    let data = a_data.ln();
    t.push(data, [a, usize::MAX], [1.0 / a_data, 0.0])
}

pub fn relu(t: &mut Tape, a: V) -> V {
    let x = t.data[a];
    let d = if x > 0.0 { 1.0 } else { 0.0 };
    t.push(x.max(0.0), [a, usize::MAX], [d, 0.0])
}

pub fn backward(t: &mut Tape, root: V) {
    t.grad[root] = 1.0;

    for i in (0..=root).rev() {
        let g = t.grad[i];
        if g == 0.0 {
            continue;
        }

        let ch = t.children[i];
        let lg = t.local_grads[i];

        if ch[0] != usize::MAX {
            t.grad[ch[0]] += lg[0] * g;
            if ch[1] != usize::MAX {
                t.grad[ch[1]] += lg[1] * g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_math() {
        let mut t = Tape::new();

        let a = t.val(2.0);
        let b = t.val(-3.0);
        let c = t.val(10.0);

        let ab = mul(&mut t, a, b);
        let z = add(&mut t, ab, c);

        assert_eq!(t.data[z], 4.0);

        backward(&mut t, z);

        assert_eq!(t.grad[a], -3.0);
        assert_eq!(t.grad[b], 2.0);
        assert_eq!(t.grad[c], 1.0);
    }

    #[test]
    fn test_diamond_problem() {
        let mut t = Tape::new();

        let x = t.val(3.0);
        let y = add(&mut t, x, x);

        backward(&mut t, y);

        assert_eq!(t.data[y], 6.0);
        assert_eq!(t.grad[x], 2.0);
    }

    #[test]
    fn test_complex_expression() {
        let mut t = Tape::new();

        let a = t.val(2.0);
        let b = t.val(5.0);

        let a_sq = pow(&mut t, a, 2.0);
        let ab = mul(&mut t, a, b);
        let f = add(&mut t, ab, a_sq);

        backward(&mut t, f);

        assert_eq!(t.data[f], 14.0);
        assert_eq!(t.grad[a], 9.0);
        assert_eq!(t.grad[b], 2.0);
    }

    #[test]
    fn test_nonlinear_relu() {
        let mut t = Tape::new();

        let x1 = t.val(0.5);
        let x2 = t.val(-1.0);

        let r1 = relu(&mut t, x1);
        let r2 = relu(&mut t, x2);

        // Combine them into a single output so one backward call handles both
        let total = add(&mut t, r1, r2);

        backward(&mut t, total);

        assert_eq!(t.grad[x1], 1.0); // relu(0.5) -> grad 1.0
        assert_eq!(t.grad[x2], 0.0); // relu(-1.0) -> grad 0.0

        // Test resetting
        t.zero_grad();
        assert_eq!(t.grad[x1], 0.0);
    }
}
