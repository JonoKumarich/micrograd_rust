use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

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

    fn set_data(&mut self, data: f32) {
        self.0.borrow_mut().data = data
    }

    fn get_grad(&self) -> f32 {
        self.0.as_ref().borrow().grad
    }

    fn set_grad(&mut self, grad: f32) {
        self.0.borrow_mut().grad = grad
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
            grad: 1.0,
            children: vec![self.clone()],
            op: Some(Operation::Tanh),
        })))
    }

    fn update_gradients(&self) {
        // We need to have the root not gradient set already...
        let mut children = self.get_children();
        match self.get_operation() {
            // a + b = c
            // da/dc = 1 -> a.grad += 1.0 * c.grad
            // db/dc = 1 -> b.grad += 1.0 * c.grad
            Some(Operation::Add) => {
                children[0].set_grad(self.get_grad());
                children[1].set_grad(self.get_grad());
            }
            // a * b = c
            // da/dc = b -> a.grad += b.data * c.grad
            // db/dc = a -> b.grad += a.data * c.grad
            Some(Operation::Mul) => {
                let data = self.get_children();
                children[0].set_grad(data[1].get_data() * self.get_grad());
                children[1].set_grad(data[0].get_data() * self.get_grad());
            }
            // tanh(a) = b
            // da/db = 1 - tanh(a)**2 = 1 - b**2
            Some(Operation::Tanh) => {
                children[0].set_grad(1.0 - self.get_data().powf(2.0));
            }
            None => (),
        }

        for child in children {
            child.update_gradients()
        }
    }

    pub fn backprop(&mut self) {
        // First gradient always 1.0 (derivate with itself)
        self.set_grad(1.0);
        self.update_gradients();
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

pub fn search_graph_order(value: &Value, nodes: &mut Vec<Value>) {
    for child in value.get_children() {
        search_graph_order(&child, nodes);
    }
    nodes.push(value.clone());
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn add_two_values() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);

        assert_eq!((a + b).get_data(), 5.0)
    }

    #[test]
    fn mul_two_values() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);

        assert_eq!((a * b).get_data(), 6.0)
    }

    #[test]
    fn compute_tanh() {
        let a = Value::new(1.0);
        assert_eq!(a.tanh().get_data(), a.get_data().tanh())
    }
}
