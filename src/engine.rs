use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

#[derive(Clone)]
struct ValueData {
    data: f32,
    grad: f32,
    children: Vec<Value>,
    op: Option<Operation>,
}

#[derive(Clone)]
enum Operation {
    Add,
    Mul,
    Tanh,
    Exp,
    Pow,
}

impl Value {
    pub fn new(value: f32) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data: value,
            grad: 0.0,
            children: Vec::new(),
            op: None,
        })))
    }

    fn get_data(&self) -> f32 {
        self.0.as_ref().borrow().data
    }

    fn get_grad(&self) -> f32 {
        self.0.as_ref().borrow().grad
    }

    pub fn set_grad(&mut self, grad: f32) {
        self.0.borrow_mut().grad = grad
    }

    fn add_grad(&mut self, grad: f32) {
        self.0.borrow_mut().grad += grad
    }

    fn get_operation(&self) -> Option<Operation> {
        self.0.as_ref().borrow().op.clone()
    }

    pub fn get_children(&self) -> Vec<Self> {
        self.0.as_ref().borrow().children.clone()
    }

    pub fn tanh(&self) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data().tanh(),
            grad: 0.0,
            children: vec![self.clone()],
            op: Some(Operation::Tanh),
        })))
    }

    pub fn exp(&self) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data().exp(),
            grad: 0.0,
            children: vec![self.clone()],
            op: Some(Operation::Tanh),
        })))
    }

    pub fn powf(&self, x: f32) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data().powf(x),
            grad: 0.0,
            children: vec![self.clone(), Value::new(x)],
            op: Some(Operation::Pow),
        })))
    }

    fn update_gradients(&self) {
        // We need to have the root not gradient set already...
        let mut children = self.get_children();
        match self.get_operation() {
            // a + b = c
            // dc/da = 1 -> a.grad += 1.0 * c.grad
            // dc/db = 1 -> b.grad += 1.0 * c.grad
            Some(Operation::Add) => {
                children[0].add_grad(self.get_grad());
                children[1].add_grad(self.get_grad());
            }
            // a * b = c
            // dc/da = b -> a.grad += b.data * c.grad
            // dc/db = a -> b.grad += a.data * c.grad
            Some(Operation::Mul) => {
                let data = self.get_children();
                children[0].add_grad(data[1].get_data() * self.get_grad());
                children[1].add_grad(data[0].get_data() * self.get_grad());
            }
            // tanh(a) = b
            // db/da = 1 - tanh(a)**2 = 1 - b**2
            Some(Operation::Tanh) => {
                children[0].add_grad(1.0 - self.get_data().powf(2.0));
            }
            // exp(a) = b
            // db/da = f'(a) * exp(a) = a.grad * a.data
            Some(Operation::Exp) => children[0].add_grad(self.get_data() * self.get_grad()),
            // a**k = b
            // db/da = k * a**(k - 1) = k.data * a.data.powf(k.data - 1) * k.grad
            Some(Operation::Pow) => {
                let x = children[1].clone();
                let val = children[0].clone();
                children[0].add_grad(
                    x.get_data() * val.get_data().powf(x.get_data() - 1.0) * self.get_grad(),
                )
            }
            None => (),
        }

        for child in children {
            child.update_gradients()
        }
    }

    pub fn backward(&mut self) {
        // First gradient always 1.0 (derivate with itself)
        self.set_grad(1.0);
        self.update_gradients();
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() + rhs.get_data(),
            grad: 0.0,
            children: vec![self, rhs],
            op: Some(Operation::Add),
        })))
    }
}

impl Add<f32> for Value {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() + rhs,
            grad: 0.0,
            children: vec![self, Value::new(rhs)],
            op: Some(Operation::Add),
        })))
    }
}

impl Add<Value> for f32 {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: rhs.get_data() + self,
            grad: 0.0,
            children: vec![rhs, Value::new(self)],
            op: Some(Operation::Add),
        })))
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        self.to_owned() + rhs.to_owned()
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() - rhs.get_data(),
            grad: 0.0,
            children: vec![self, rhs],
            op: Some(Operation::Add),
        })))
    }
}

impl Sub<f32> for Value {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() - rhs,
            grad: 0.0,
            children: vec![self, Value::new(rhs)],
            op: Some(Operation::Add),
        })))
    }
}

impl Sub<Value> for f32 {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: self - rhs.get_data(),
            grad: 0.0,
            children: vec![rhs, Value::new(self)],
            op: Some(Operation::Add),
        })))
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self.to_owned() - rhs.to_owned()
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() * rhs.get_data(),
            grad: 0.0,
            children: vec![self, rhs],
            op: Some(Operation::Mul),
        })))
    }
}

impl Mul<f32> for Value {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() * rhs,
            grad: 0.0,
            children: vec![self, Value::new(rhs)],
            op: Some(Operation::Add),
        })))
    }
}

impl Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: rhs.get_data() * self,
            grad: 0.0,
            children: vec![rhs, Value::new(self)],
            op: Some(Operation::Add),
        })))
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        self.to_owned() * rhs.to_owned()
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() / rhs.get_data(),
            grad: 0.0,
            children: vec![self, rhs],
            op: Some(Operation::Mul),
        })))
    }
}

impl Div<f32> for Value {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() / rhs,
            grad: 0.0,
            children: vec![self, Value::new(rhs)],
            op: Some(Operation::Add),
        })))
    }
}

impl Div<Value> for f32 {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: rhs.get_data() / self,
            grad: 0.0,
            children: vec![rhs, Value::new(self)],
            op: Some(Operation::Add),
        })))
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        self.to_owned() / rhs.to_owned()
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Value={}, Grad={}]", self.get_data(), self.get_grad())
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}
