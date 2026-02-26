use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

// Generic struct with lifetime and trait bounds
#[derive(Debug)]
pub struct Container<T: Display + ?Sized> {
    value: T,
    metadata: HashMap<String, String>,
}

// Generic function with lifetime parameters
pub fn print_container<'a, T: Display + 'a>(container: &'a Container<T>) {
    println!("Container: {{}}", container.value);
}

// Macro definition
macro_rules! make_container {
    ($name:ident, $type:ty) => {
        pub struct $name {
            inner: $type,
        }
    };
}

// Macro invocation
make_container!(IntContainer, i32);

// Trait definition with associated types
pub trait Processor {
    type Input;
    type Output;

    fn process(&self, input: Self::Input) -> Self::Output;
}

// Trait implementation
impl Processor for IntContainer {
    type Input = i32;
    type Output = String;

    fn process(&self, input: i32) -> String {
        format!("Processed: {{}}", input + self.inner)
    }
}

// Another trait with generic methods
pub trait Converter<T, U> {
    fn convert(&self, item: T) -> U;
}

// Enum with generics
#[derive(Debug)]
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Pattern matching examples
pub fn match_example(result: Result<i32, String>) -> String {
    match result {
        Result::Ok(value) => format!("Success: {{}}", value),
        Result::Err(err) => format!("Error: {{}}", err),
    }
}

// Complex pattern matching with guards
pub fn complex_match(value: i32) -> &'static str {
    match value {
        0 => "zero",
        1 | 2 | 3 => "small",
        n if n > 10 => "large",
        _ => "medium",
    }
}

// Struct with tuple struct pattern
pub struct Point(i32, i32);

pub fn tuple_match(point: Point) -> &'static str {
    match point {
        Point(0, 0) => "origin",
        Point(x, 0) => "on x-axis",
        Point(0, y) => "on y-axis",
        Point(_, _) => "elsewhere",
    }
}

// Function with where clause
pub fn where_example<T, U>(t: T, u: U) -> String
where
    T: Display,
    U: Display,
{
    format!("{{}} and {{}}", t, u)
}

// Associated function
impl<T> Container<T> {
    pub fn new(value: T) -> Self {
        Container {
            value,
            metadata: HashMap::new(),
        }
    }
}

// Associated constant
impl<T> Container<T> {
    pub const DEFAULT_CAPACITY: usize = 100;
}

// Test module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container() {
        let container = Container::new(42);
        assert_eq!(container.value, 42);
    }

    #[test]
    fn test_macro() {
        let int_container = IntContainer { inner: 10 };
        assert_eq!(int_container.inner, 10);
    }

    #[test]
    fn test_trait_impl() {
        let int_container = IntContainer { inner: 5 };
        let result = int_container.process(10);
        assert_eq!(result, "Processed: 15");
    }
}
