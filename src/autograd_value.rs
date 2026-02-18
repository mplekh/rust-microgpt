// -----------------------------
// Autograd Value
// "Micrograd-style" implementation of reverse-mode automatic differentiation
// -----------------------------

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub type V = Rc<RefCell<Value>>;

#[derive(Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    children: Vec<V>,
    local_grads: Vec<f64>,
}

pub fn val(data: f64) -> V {
    Rc::new(RefCell::new(Value {
        data,
        grad: 0.0,
        children: vec![],
        local_grads: vec![],
    }))
}

pub fn add(a: &V, b: &V) -> V {
    let out = val(a.borrow().data + b.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.children = vec![a.clone(), b.clone()];
        o.local_grads = vec![1.0, 1.0];
    }
    out
}

pub fn mul(a: &V, b: &V) -> V {
    let out = val(a.borrow().data * b.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.children = vec![a.clone(), b.clone()];
        o.local_grads = vec![b.borrow().data, a.borrow().data];
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
        o.children = vec![a.clone()];
        o.local_grads = vec![p * a.borrow().data.powf(p - 1.0)];
    }
    out
}

pub fn exp(a: &V) -> V {
    let e = a.borrow().data.exp();
    let out = val(e);
    {
        let mut o = out.borrow_mut();
        o.children = vec![a.clone()];
        o.local_grads = vec![e];
    }
    out
}

pub fn log(a: &V) -> V {
    let out = val(a.borrow().data.ln());
    {
        let mut o = out.borrow_mut();
        o.children = vec![a.clone()];
        o.local_grads = vec![1.0 / a.borrow().data];
    }
    out
}

pub fn relu(a: &V) -> V {
    let d = if a.borrow().data > 0.0 { 1.0 } else { 0.0 };
    let out = val(a.borrow().data.max(0.0));
    {
        let mut o = out.borrow_mut();
        o.children = vec![a.clone()];
        o.local_grads = vec![d];
    }
    out
}

pub fn backward(root: &V) {
    let mut topo = vec![];
    let mut visited = HashSet::new();

    fn build(v: &V, topo: &mut Vec<V>, visited: &mut HashSet<usize>) {
        let addr = Rc::as_ptr(v) as usize;
        if !visited.contains(&addr) {
            visited.insert(addr);
            for c in &v.borrow().children {
                build(c, topo, visited);
            }
            topo.push(v.clone());
        }
    }

    build(root, &mut topo, &mut visited);

    root.borrow_mut().grad = 1.0;

    for v in topo.into_iter().rev() {
        let grad = v.borrow().grad;
        let children = v.borrow().children.clone();
        let locals = v.borrow().local_grads.clone();
        for (c, lg) in children.iter().zip(locals.iter()) {
            c.borrow_mut().grad += lg * grad;
        }
    }
}
