mod engine;
use engine::Value;

mod network;
use network::MLP;

fn main() {
    let nn = MLP::new(3, &vec![4, 4, 1]);

    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];

    let y = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    let ypred: Vec<Vec<Value>> = xs.iter().map(|x| nn.forward(&x)).collect();

    println!("{:?}", ypred);
}
