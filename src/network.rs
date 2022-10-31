use crate::engine::Value;
use rand::distributions::{Distribution, Uniform};

#[derive(Debug)]
struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    fn new(nin: u32) -> Self {
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

    fn forward(&self, x: &Vec<Value>) -> Value {
        self.w
            .iter()
            .zip(x.iter())
            .map(|(xi, wi)| xi * wi)
            .fold(self.b.clone(), |a, b| a + b)
            .tanh()
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: u32, nout: u32) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: u32, nout: &Vec<u32>) -> Self {
        let mut sz = nout.clone();
        sz.insert(0, nin);

        Self {
            layers: (0..nout.len())
                .map(|i| Layer::new(sz[i], sz[i + 1]))
                .collect(),
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let activations: Vec<Vec<Value>> = self.layers.iter().map(|l| l.forward(x)).collect();
        activations[activations.len() - 1].clone()
    }
}
