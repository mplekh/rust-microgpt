use std::f32;
use std::ptr;

pub type DataT = f32;
pub type GradT = f32;

pub struct Matrix {
    pub data_start: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { data_start: 0, rows, cols }
    }

    pub fn at(&self, i: usize, j: usize) -> usize {
        self.data_start + i * self.cols + j
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum Op {
    Const,
    SubConst,
    Relu,
    InvLog,
    InvSqrt,
    Exp,
    Add,
    Mul,
    Div,
    MulAdd,
}

#[derive(Clone)]
struct Children {
    a: u32,
    b: u32,
    c: u32
}

pub struct Tape {
    pub data: Vec<DataT>,
    pub grad: Vec<GradT>,
    child: Vec<Children>,
    op: Vec<Op>,
    size: usize,
    cap: usize,
}

impl Tape {
pub fn new(n: usize) -> Self {
        Self {
            data: vec![0.0; n],
            grad: vec![0.0; n],
            child: vec![Children{a: 0, b: 0, c: 0}; n],
            op: vec![Op::Const; n],
            size: 0,
            cap: n,
        }
    }

    #[inline(always)]
    fn grow(&mut self, new_cap: usize) {
        self.data.resize(new_cap, 0.0);
        self.grad.resize(new_cap, 0.0);
        self.child.resize(new_cap, Children{a: 0, b: 0, c: 0});
        self.op.resize(new_cap, Op::Const);
        self.cap = new_cap;
    }

    #[inline(always)]
    fn ensure(&mut self) {
        if self.size >= self.cap {
            self.grow(self.cap * 2);
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.size }

    #[inline(always)]
    pub fn truncate(&mut self, n: usize) { self.size = n; }

#[inline(always)]
    pub fn push_const(&mut self, d: DataT) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = d;
        self.op[i] = Op::Const;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn relu(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a].max(0.0);
        self.child[i].a = a as u32;
        self.op[i] = Op::Relu;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn inv_log(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = -self.data[a].ln();
        self.child[i].a = a as u32;
        self.op[i] = Op::InvLog;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn inv_sqrt(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        let val = (self.data[a] + 1e-5).powf(-0.5);
        self.data[i] = val;
        self.child[i].a = a as u32;
        self.op[i] = Op::InvSqrt;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn exp(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        let val = self.data[a].exp();
        self.data[i] = val;
        self.child[i].a = a as u32;
        self.op[i] = Op::Exp;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn sub_const(&mut self, a: usize, c: DataT) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] - c;
        self.child[i].a = a as u32;
        self.op[i] = Op::SubConst;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] + self.data[b];
        self.child[i].a = a as u32;
        self.child[i].b = b as u32;
        self.op[i] = Op::Add;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.ensure();
        let i = self.size;
        unsafe {
            let base_data = self.data.as_mut_ptr();
            let base_op = self.op.as_mut_ptr();
            let base_c = self.child.as_mut_ptr();
            let va = *base_data.add(a);
            let vb = *base_data.add(b);
            *base_data.add(i) = va * vb;
            *base_op.add(i) = Op::Mul;
            *base_c.add(i) = Children{a: a as u32, b: b as u32, c: 0u32};
        }
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn mul_const(&mut self, a: usize, c: DataT) -> usize {
        let i_c = self.push_const(c);
        self.mul(a, i_c)
    }

    #[inline(always)]
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] / self.data[b];
        self.child[i].a = a as u32;
        self.child[i].b = b as u32;
        self.op[i] = Op::Div;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn div_const(&mut self, a: usize, c: DataT) -> usize {
        let i_c = self.push_const(c);
        self.div(a, i_c)
    }

    #[inline(always)]
    pub fn mul_add(&mut self, a: usize, b: usize, c: usize) -> usize {
        self.ensure();
        self.mul_add_internal(a, b, c)
    }

    #[inline(always)]
    fn mul_add_internal(&mut self, a: usize, b: usize, c: usize) -> usize {
        let i = self.size;
        self.size += 1;

        // Using raw pointers for maximum speed in the hot loop
        unsafe {
            let base_data = self.data.as_mut_ptr();
            let base_op = self.op.as_mut_ptr();
            let base_c = self.child.as_mut_ptr();

            // Fetch values from data pointer
            let va = *base_data.add(a);
            let vb = *base_data.add(b);
            let product = va * vb;

            // i1: The Multiplication Node
            *base_data.add(i) = product + *base_data.add(c);
            *base_op.add(i) = Op::MulAdd;
            *base_c.add(i) = Children{a: a as u32, b: b as u32, c: c as u32};
        }
        i
    }

    pub fn softmax(&mut self, out: &mut [usize], logits: &[usize]) {
        let mut max_val = DataT::MIN;
        for &idx in logits {
            if self.data[idx] > max_val { max_val = self.data[idx]; }
        }

        for i in 0..logits.len() {
            let sub = self.sub_const(logits[i], max_val);
            out[i] = self.exp(sub);
        }

        let mut sum_exponents = out[0];
        for i in 1..logits.len() {
            sum_exponents = self.add(sum_exponents, out[i]);
        }

        for i in 0..logits.len() {
            out[i] = self.div(out[i], sum_exponents);
        }
    }

    pub fn rmsnorm(&mut self, out: &mut [usize], x: &[usize]) {
        let mut sum_squares = self.mul(x[0], x[0]);
        for i in 1..x.len() {
            sum_squares = self.mul_add(x[i], x[i], sum_squares);
        }
        let mean_square = self.div_const(sum_squares, x.len() as DataT);
        let scale = self.inv_sqrt(mean_square);
        for i in 0..x.len() {
            out[i] = self.mul(x[i], scale);
        }
    }

    pub fn linear(&mut self, out: &mut [usize], x: &[usize], w: &Matrix) {
        if self.size + w.rows * w.cols * 2 >= self.cap {
            self.grow(self.cap * 2);
        }
        for i in 0..w.rows {
            let w_at_row_i = w.at(i, 0);
            let mut sum = self.mul(w_at_row_i, x[0]);
            for j in 1..w.cols {
                sum = self.mul_add_internal(w_at_row_i + j, x[j], sum);
            }
            out[i] = sum;
        }
    }

    #[inline(never)]
    pub fn backward(&mut self, loss_idx: usize) {
        // Equivalent to std::memset(grad, 0, n * sizeof(grad_T))
        unsafe {
            ptr::write_bytes(self.grad.as_mut_ptr(), 0, loss_idx + 1);
        }
        self.grad[loss_idx] = 1.0;

        // Using raw pointers for maximum speed in the hot loop
        let p_op = self.op.as_ptr();
        let p_grad = self.grad.as_mut_ptr();
        let p_data = self.data.as_ptr();
        let p_c = self.child.as_ptr();

        for i in (0..=loss_idx).rev() {
            unsafe {
                let g = *p_grad.add(i);
                if g == 0.0 { continue; }

                let a = (*p_c.add(i)).a as usize;
                let b = (*p_c.add(i)).b as usize;

                let op = *p_op.add(i);
                match op {
                    Op::SubConst => {
                        *p_grad.add(a) += g;
                    }
                    Op::Add => {
                        *p_grad.add(a) += g;
                        *p_grad.add(b) += g;
                    }
                    Op::Relu => {
                        if *p_data.add(a) > 0.0 {
                            *p_grad.add(a) += g;
                        }
                    }
                    Op::InvLog => {
                        *p_grad.add(a) -= g / *p_data.add(a);
                    }
                    Op::InvSqrt => {
                        *p_grad.add(a) -= 0.5 * g * (*p_data.add(i)) / (*p_data.add(a) + 1e-5);
                    }
                    Op::Exp => {
                        *p_grad.add(a) += g * (*p_data.add(i));
                    }
                    Op::Mul => {
                        *p_grad.add(a) += g * (*p_data.add(b));
                        *p_grad.add(b) += g * (*p_data.add(a));
                    }
                    Op::Div => {
                        let d_c1 = *p_data.add(b);
                        *p_grad.add(a) += g / d_c1;
                        *p_grad.add(b) -= g * (*p_data.add(a)) / (d_c1 * d_c1);
                    }
                    Op::MulAdd => {
                        let c = (*p_c.add(i)).c as usize;
                        *p_grad.add(a) += g * (*p_data.add(b));
                        *p_grad.add(b) += g * (*p_data.add(a));
                        *p_grad.add(c) += g;
                    }
                    _ => {}
                }
            }
        }
    }
}
