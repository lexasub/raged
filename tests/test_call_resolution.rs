// Rust test file for enhanced call resolution testing
// Tests type inference, virtual method dispatch, lambda/closure extraction, and cross-file symbol resolution
use std::collections::HashMap;
use std::sync::Arc;
use std::fmt::Debug;

// Base trait for virtual method dispatch testing
trait Animal {
    fn speak(&self);
}

struct Dog;

impl Animal for Dog {
    fn speak(&self) {
        println!("Woof!");
    }
}

struct Cat;

impl Animal for Cat {
    fn speak(&self) {
        println!("Meow!");
    }
}

// Struct for testing lambda/closure extraction
struct MathOperations;

impl MathOperations {
    pub fn test_lambdas(&self) {
        let x = 10;
        
        // Simple lambda
        let add = |a: i32, b: i32| a + b;
        let result1 = add(5, 3);
        
        // Lambda with capture
        let multiply_by_x = |a: i32| a * x;
        let result2 = multiply_by_x(5);
        
        // Lambda with move capture
        let counter = std::cell::RefCell::new(0);
        let increment = move || {
            let mut val = counter.borrow_mut();
            *val += 1;
            *val
        };
        let result3 = increment();
        
        // Lambda in collection
        let mut operations: Vec<Box<dyn Fn(i32) -> i32>> = Vec::new();
        operations.push(Box::new(|a| a * 2));
        operations.push(Box::new(|a| a + 10));
        
        let result4 = operations[0](5);
        let result5 = operations[1](5);
    }
}

// Struct for testing cross-file symbol resolution
struct FileResolver;

impl FileResolver {
    pub fn resolve_symbols(&self) {
        // This will test cross-file resolution
        let external = ExternalClass::new();
        external.external_method();
        
        // Test with generics
        let int_template = TemplateClass::new();
        int_template.do_something();
        
        let string_template = TemplateClass::new();
        string_template.do_something();
    }
}

// External class defined in another file (for cross-file testing)
struct ExternalClass;

impl ExternalClass {
    pub fn new() -> Self {
        ExternalClass
    }
    
    pub fn external_method(&self) {
        println!("External method called");
    }
    
    // Static method for testing
    pub fn static_external_method() {
        println!("Static external method called");
    }
    
    // Nested struct for testing
    pub struct NestedExternal;
    
    impl NestedExternal {
        pub fn nested_method(&self) {
            println!("Nested external method called");
        }
    }
}

// Generic struct for testing generic instantiation
struct TemplateClass<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T> TemplateClass<T> {
    pub fn new() -> Self {
        TemplateClass {
            phantom: std::marker::PhantomData,
        }
    }
    
    pub fn do_something(&self) {
        println!("Template value: {:?}", std::any::type_name::<T>());
    }
    
    // Nested generic struct
    pub struct NestedTemplate<U> {
        phantom: std::marker::PhantomData<U>,
    }
    
    impl<U> NestedTemplate<U> {
        pub fn new() -> Self {
            NestedTemplate {
                phantom: std::marker::PhantomData,
            }
        }
        
        pub fn nested_do_something(&self) {
            println!("Nested template value: {:?}", std::any::type_name::<U>());
        }
    }
}

// Function for testing function overloading
fn overloaded_function(x: i32) {
    println!("Overloaded with i32: {}", x);
}

fn overloaded_function(x: f64) {
    println!("Overloaded with f64: {}", x);
}

fn overloaded_function(x: &str) {
    println!("Overloaded with &str: {}", x);
}

// Function for testing variadic templates
trait VariadicFunction {
    fn variadic_function(&self);
}

impl VariadicFunction for () {
    fn variadic_function(&self) {
        println!("Variadic function called with 0 arguments");
    }
}

impl<A> VariadicFunction for (A,) {
    fn variadic_function(&self) {
        println!("Variadic function called with 1 argument");
    }
}

impl<A, B, C> VariadicFunction for (A, B, C) {
    fn variadic_function(&self) {
        println!("Variadic function called with 3 arguments");
    }
}

// Lambda factory function for testing
fn create_multiplier(factor: i32) -> impl Fn(i32) -> i32 {
    move |x| x * factor
}

// Main function to test all features
fn main() {
    // Test virtual method dispatch
    let dog = Dog;
    let cat = Cat;
    
    dog.speak();
    cat.speak();
    
    // Test lambda/closure extraction
    let math_ops = MathOperations;
    math_ops.test_lambdas();
    
    // Test cross-file symbol resolution
    let resolver = FileResolver;
    resolver.resolve_symbols();
    
    // Test function overloading
    overloaded_function(42);
    overloaded_function(3.14);
    overloaded_function("Hello");
    
    // Test variadic templates
    let unit: () = ();
    unit.variadic_function();
    
    let one = (1,);
    one.variadic_function();
    
    let three = (1, 2.5, "test");
    three.variadic_function();
    
    // Test lambda factory
    let multiply_by_3 = create_multiplier(3);
    let result = multiply_by_3(5);
    println!("Lambda factory result: {}", result);
    
    // Test async/await
    async fn async_test() {
        println!("Async test started");
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        println!("Async test completed");
    }
    
    // Run async test
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async_test());
    
    // Test trait objects and dynamic dispatch
    let animals: Vec<Box<dyn Animal>> = vec![
        Box::new(Dog),
        Box::new(Cat),
    ];
    
    for animal in animals {
        animal.speak();
    }
    
    // Test generic functions
    fn generic_function<T: Debug>(value: T) {
        println!("Generic function called with: {:?}", value);
    }
    
    generic_function(42);
    generic_function("test");
    generic_function(vec![1, 2, 3]);
}