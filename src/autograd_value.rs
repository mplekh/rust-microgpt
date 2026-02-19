// -----------------------------
// Autograd Value
// "Micrograd-style" implementation of reverse-mode automatic differentiation
// -----------------------------

use std::cell::RefCell;
use std::rc::Rc;

pub type V = Rc<RefCell<Value>>;

#[derive(Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    children: [Option<V>; 2],
    local_grads: [f64; 2],
    arity: u8,
    visited: bool,
}

pub fn val(data: f64) -> V {
    Rc::new(RefCell::new(Value {
        data,
        grad: 0.0,
        children: [None, None],
        local_grads: [0.0, 0.0],
        arity: 0,
        visited: false,
    }))
}

pub fn add(a: &V, b: &V) -> V {
    let out = val(a.borrow().data + b.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.children[0] = Some(a.clone());
        o.children[1] = Some(b.clone());
        o.local_grads = [1.0, 1.0];
        o.arity = 2;
    }
    out
}

pub fn mul(a: &V, b: &V) -> V {
    let out = val(a.borrow().data * b.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.children[0] = Some(a.clone());
        o.children[1] = Some(b.clone());
        o.local_grads = [b.borrow().data, a.borrow().data];
        o.arity = 2;
    }
    out
}

pub fn neg(a: &V) -> V {
    mul(a, &val(-1.0))
}

pub fn sub(a: &V, b: &V) -> V {
    add(a, &neg(b))
}

pub fn div(a: &V, b: &V) -> V {
    let inv = pow(b, -1.0);
    mul(a, &inv)
}

pub fn pow(a: &V, p: f64) -> V {
    let out = val(a.borrow().data.powf(p));
    {
        let mut o = out.borrow_mut();
        o.children[0] = Some(a.clone());
        o.local_grads = [p * a.borrow().data.powf(p - 1.0), 0.0];
        o.arity = 1;
    }
    out
}

pub fn exp(a: &V) -> V {
    let e = a.borrow().data.exp();
    let out = val(e);
    {
        let mut o = out.borrow_mut();
        o.children[0] = Some(a.clone());
        o.local_grads = [e, 0.0];
        o.arity = 1;
    }
    out
}

pub fn log(a: &V) -> V {
    let out = val(a.borrow().data.ln());
    {
        let mut o = out.borrow_mut();
        o.children[0] = Some(a.clone());
        o.local_grads = [1.0 / a.borrow().data, 0.0];
        o.arity = 1;
    }
    out
}

pub fn relu(a: &V) -> V {
    let d = if a.borrow().data > 0.0 { 1.0 } else { 0.0 };
    let out = val(a.borrow().data.max(0.0));
    {
        let mut o = out.borrow_mut();
        o.children[0] = Some(a.clone());
        o.local_grads = [d, 0.0];
        o.arity = 1;
    }
    out
}

pub fn backward(root: &V) {
    let mut topo = vec![];

    fn build(v: &V, topo: &mut Vec<V>) {
        if v.borrow().visited {
            return;
        }

        v.borrow_mut().visited = true;

        let v_b = v.borrow();
        for i in 0..v_b.arity as usize {
            let child = v_b.children[i].as_ref().unwrap();
            build(child, topo);
        }

        topo.push(v.clone());
    }

    build(root, &mut topo);

    // Seed the gradient
    root.borrow_mut().grad = 1.0;

    // Process in reverse topological order
    for v in topo.clone().into_iter().rev() {
        let v_borrow = v.borrow();
        let grad = v_borrow.grad;

        for i in 0..v_borrow.arity as usize {
            let child = v_borrow.children[i].as_ref().unwrap();
            child.borrow_mut().grad += v_borrow.local_grads[i] * grad;
        }
    }
    for v in &topo {
        v.borrow_mut().visited = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_math() {
        // z = (a * b) + c
        let a = val(2.0);
        let b = val(-3.0);
        let c = val(10.0);

        let ab = mul(&a, &b);
        let z = add(&ab, &c);

        assert_eq!(z.borrow().data, 4.0);

        backward(&z);

        // dz/da = b = -3
        assert_eq!(a.borrow().grad, -3.0);
        // dz/db = a = 2
        assert_eq!(b.borrow().grad, 2.0);
        // dz/dc = 1
        assert_eq!(c.borrow().grad, 1.0);
    }

    // The "Diamond Problem" is the most common bug in custom autograd engines.
    // It happens when a node has multiple parents in the computational graph.
    #[test]
    fn test_diamond_problem() {
        // Testing gradient accumulation:
        // x = 3
        // y = x + x  (dy/dx should be 2)
        let x = val(3.0);
        let y = add(&x, &x);

        backward(&y);

        assert_eq!(y.borrow().data, 6.0);
        assert_eq!(x.borrow().grad, 2.0); // 1.0 + 1.0
    }

    #[test]
    fn test_complex_expression() {
        // f(a, b) = (a * b) + a^2
        // df/da = b + 2a
        let a = val(2.0);
        let b = val(5.0);

        let a_sq = pow(&a, 2.0);
        let ab = mul(&a, &b);
        let f = add(&ab, &a_sq);

        backward(&f);

        assert_eq!(f.borrow().data, 14.0);
        // df/da = 5 + 2(2) = 9
        assert_eq!(a.borrow().grad, 9.0);
        // df/db = a = 2
        assert_eq!(b.borrow().grad, 2.0);
    }

    #[test]
    fn test_nonlinear_relu() {
        let x1 = val(0.5);
        let x2 = val(-1.0);

        let r1 = relu(&x1);
        let r2 = relu(&x2);

        backward(&r1);
        backward(&r2);

        assert_eq!(x1.borrow().grad, 1.0); // Grad exists for > 0
        assert_eq!(x2.borrow().grad, 0.0); // Grad is 0 for < 0
    }
}
