mod engine;
use engine::Value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);

    let c = &a + &b;
    let d = &a + &b;

    let e = a.tanh();

    // let mut children = ;
    println!("{}", c.get_children()[0]);
    c.get_children()[0].set_grad(1.0);
    println!("{}", c.get_children()[0]);

    println!("{}", d.get_children()[0]);

    println!("{}", a)
}
