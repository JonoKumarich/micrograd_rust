mod engine;
use engine::Value;

fn main() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    let b = Value::new(6.8813735870195432);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let mut o = n.tanh();

    o.backprop();

    let n = &o.get_children()[0];
    let x1w1x2w2 = &n.get_children()[0];
    let b = &n.get_children()[0];
    let x1w1 = &x1w1x2w2.get_children()[0];
    let x2w2 = &x1w1x2w2.get_children()[1];
    let x1 = &x1w1.get_children()[0];
    let x2 = &x2w2.get_children()[0];
    let w1 = &x1w1.get_children()[1];
    let w2 = &x2w2.get_children()[1];

    println!("{:?}", n);
    println!("{:?}", x1w1x2w2);
    println!("{:?}", b);
    println!("{:?}", x1w1);
    println!("{:?}", x2w2);
    println!("{:?}", x1);
    println!("{:?}", x2);
    println!("{:?}", w1);
    println!("{:?}", w2);
}
