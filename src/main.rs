use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

fn main() {
    let a = Value::new(3.0);
    let b = Value::new(2.0);

    // let c = a + b;
    println!("{:?}", a)
}

#[derive(Debug, Clone)]
struct Value(Rc<RefCell<ValueData>>);

#[derive(Debug)]
struct ValueData {
    data: f32,
    grad: f32,
    children: Vec<Value>,
    op: Option<Operation>,
}

#[derive(Debug)]
enum Operation {
    Add,
    Mul,
}

impl Value {
    fn new(value: f32) -> Self {
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
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() + rhs.get_data(),
            grad: 0.0,
            children: vec![self, rhs],
            op: Some(Operation::Add),
        })))
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data: self.get_data() * rhs.get_data(),
            grad: 0.0,
            children: vec![self, rhs],
            op: Some(Operation::Mul),
        })))
    }
}
