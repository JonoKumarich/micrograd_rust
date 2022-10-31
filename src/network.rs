use crate::engine::Value;
use rand::distributions::{Distribution, Uniform};

#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: u32) -> Self {
        let between = Uniform::from(-1_f32..1_f32);
        let mut rng = rand::thread_rng();

        let w: Vec<Value> = (0..nin)
            .map(|_| Value::new(between.sample(&mut rng)))
            .collect();

        Self {
            w,
            b: Value::new(between.sample(&mut rng) as f32),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Value {
        self.w
            .iter()
            .zip(x.iter())
            .map(|(xi, wi)| xi * wi)
            .fold(self.b.clone(), |a, b| a + b)
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: u32, nout: u32) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x.clone())).collect()
    }
}

struct MLP {}
