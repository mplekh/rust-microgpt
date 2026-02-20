// -----------------------------
// Autograd Value
// "Micrograd-style" implementation of reverse-mode automatic differentiation
// -----------------------------

pub type V = usize;

#[derive(Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    children: [usize; 2],
    local_grads: [f64; 2],
    arity: u8,
}

pub struct Tape {
    pub nodes: Vec<Value>,
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn val(&mut self, data: f64) -> V {
        let id = self.nodes.len();
        self.nodes.push(Value {
            data,
            grad: 0.0,
            children: [0, 0],
            local_grads: [0.0, 0.0],
            arity: 0,
        });
        id
    }

    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            node.grad = 0.0;
        }
    }
}

pub fn add(t: &mut Tape, a: V, b: V) -> V {
    let data = t.nodes[a].data + t.nodes[b].data;

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, b],
        local_grads: [1.0, 1.0],
        arity: 2,
    });

    id
}

pub fn mul(t: &mut Tape, a: V, b: V) -> V {
    let data = t.nodes[a].data * t.nodes[b].data;

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, b],
        local_grads: [t.nodes[b].data, t.nodes[a].data],
        arity: 2,
    });

    id
}

pub fn neg(t: &mut Tape, a: V) -> V {
    let data = -t.nodes[a].data;

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, usize::MAX],
        local_grads: [-1.0, 0.0],
        arity: 1,
    });

    id
}

pub fn sub(t: &mut Tape, a: V, b: V) -> V {
    let data = t.nodes[a].data - t.nodes[b].data;

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, b],
        local_grads: [1.0, -1.0],
        arity: 2,
    });

    id
}

pub fn div(t: &mut Tape, a: V, b: V) -> V {
    let a_data = t.nodes[a].data;
    let b_data = t.nodes[b].data;

    let data = a_data / b_data;

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, b],
        local_grads: [1.0 / b_data, -a_data / (b_data * b_data)],
        arity: 2,
    });

    id
}

pub fn mul_const(t: &mut Tape, a: V, c: f64) -> V {
    let data = t.nodes[a].data * c;

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, usize::MAX],
        local_grads: [c, 0.0],
        arity: 1,
    });

    id
}

pub fn pow(t: &mut Tape, a: V, p: f64) -> V {
    let a_data = t.nodes[a].data;
    let data = a_data.powf(p);

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, usize::MAX],
        local_grads: [p * a_data.powf(p - 1.0), 0.0],
        arity: 1,
    });

    id
}

pub fn exp(t: &mut Tape, a: V) -> V {
    let e = t.nodes[a].data.exp();

    let id = t.nodes.len();
    t.nodes.push(Value {
        data: e,
        grad: 0.0,
        children: [a, usize::MAX],
        local_grads: [e, 0.0],
        arity: 1,
    });

    id
}

pub fn log(t: &mut Tape, a: V) -> V {
    let a_data = t.nodes[a].data;
    let data = a_data.ln();

    let id = t.nodes.len();
    t.nodes.push(Value {
        data,
        grad: 0.0,
        children: [a, usize::MAX],
        local_grads: [1.0 / a_data, 0.0],
        arity: 1,
    });

    id
}

pub fn relu(t: &mut Tape, a: V) -> V {
    let x = t.nodes[a].data;
    let d = if x > 0.0 { 1.0 } else { 0.0 };

    let id = t.nodes.len();
    t.nodes.push(Value {
        data: x.max(0.0),
        grad: 0.0,
        children: [a, usize::MAX],
        local_grads: [d, 0.0],
        arity: 1,
    });

    id
}

pub fn backward(t: &mut Tape, root: V) {
    t.nodes[root].grad = 1.0;

    // Because nodes are appended linearly, the reverse order
    // is guaranteed to be a valid topological sort.
    for i in (0..=root).rev() {
        let grad = t.nodes[i].grad;
        if grad == 0.0 { continue; } // skip nodes with no contribution

        let arity = t.nodes[i].arity as usize;
        for j in 0..arity {
            let child = t.nodes[i].children[j];
            let lg = t.nodes[i].local_grads[j];
            t.nodes[child].grad += lg * grad;
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

        assert_eq!(t.nodes[z].data, 4.0);

        backward(&mut t, z);

        assert_eq!(t.nodes[a].grad, -3.0);
        assert_eq!(t.nodes[b].grad, 2.0);
        assert_eq!(t.nodes[c].grad, 1.0);
    }

    #[test]
    fn test_diamond_problem() {
        let mut t = Tape::new();

        let x = t.val(3.0);
        let y = add(&mut t, x, x);

        backward(&mut t, y);

        assert_eq!(t.nodes[y].data, 6.0);
        assert_eq!(t.nodes[x].grad, 2.0);
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

        assert_eq!(t.nodes[f].data, 14.0);
        assert_eq!(t.nodes[a].grad, 9.0);
        assert_eq!(t.nodes[b].grad, 2.0);
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

        assert_eq!(t.nodes[x1].grad, 1.0); // relu(0.5) -> grad 1.0
        assert_eq!(t.nodes[x2].grad, 0.0); // relu(-1.0) -> grad 0.0

        // Test resetting
        t.zero_grad();
        assert_eq!(t.nodes[x1].grad, 0.0);
    }

}
