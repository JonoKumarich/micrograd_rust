mod engine;
use engine::Value;

mod network;
use network::Neuron;

fn main() {
    let n = Neuron::new(5);

    println!("{:?}", n);
}
