//! Rust code example for demonstrating block extraction.
//!
//! Contains:
//! - Functions with if/for/while blocks
//! - Closures with captured variables
//! - Try blocks (Result/Option)
//! - Match expressions

use std::collections::HashMap;

/// Processes a vector of numbers with various blocks
fn process_data(items: Vec<i32>, threshold: i32) -> Vec<i32> {
    let mut result = Vec::new();

    // IF block
    if items.is_empty() {
        return result;
    }

    // FOR block with nested IF
    for item in items {
        if item > threshold {
            result.push(item * 2);
        } else if item == threshold {
            result.push(item);
        // WHILE block inside FOR
        } else {
            let mut count = 0;
            while count < 3 {
                result.push(item + count);
                count += 1;
            }
        }
    }

    result
}

/// Creates a closure with a captured variable
fn create_multiplier(factor: i32) -> impl Fn(i32) -> i32 {
    // Closure with captured variable 'factor'
    move |x| x * factor
}

/// Division with error handling via Result
fn safe_divide(a: i32, b: i32) -> Result<f64, String> {
    // TRY block via ? operator
    if b == 0 {
        return Err("Division by zero".to_string());
    }

    let result = a as f64 / b as f64;
    Ok(result)
}

/// Function with match expression
fn classify_number(value: i32) -> &'static str {
    // MATCH block
    match value {
        n if n < 0 => "negative",
        0 => "zero",
        n if n > 0 && n <= 10 => "small positive",
        _ => "large positive",
    }
}

/// Function with multiple nested blocks
fn complex_function(data: Vec<i32>) -> i32 {
    let mut total = 0;

    // Level 1: FOR
    for (i, &value) in data.iter().enumerate() {
        // Level 2: IF
        if value > 0 {
            // Level 3: nested FOR
            for j in 0..value {
                // Level 4: IF inside nested FOR
                if j % 2 == 0 {
                    total += j;
                } else {
                    // Closure inside function
                    let transform = |x: i32| x * 2 + i as i32;
                    total += transform(j);
                }
            }
        // Level 2: ELSE with WHILE
        } else {
            let mut count = value.abs();
            // Level 3: WHILE
            while count > 0 {
                total -= 1;
                count -= 1;
            }
        }
    }

    total
}

/// Function with loop and label
fn labeled_loop_example(data: Vec<Vec<i32>>) -> Option<i32> {
    // LOOP with label
    'outer: for inner_vec in data {
        for &value in &inner_vec {
            if value > 100 {
                // Break from outer loop
                break 'outer;
            }
            if value == 42 {
                // Return from function
                return Some(value);
            }
        }
    }
    None
}

/// Global closures
fn main() {
    // Closure with capture
    let multiplier = 10;
    let scale = |x: i32| x * multiplier;

    // Usage examples
    let data = vec![1, 5, 10, 15, 20];

    // Call function with blocks
    let result = process_data(data.clone(), 10);
    println!("Processed: {:?}", result);

    // Closure with captured variable
    let doubler = create_multiplier(2);
    println!("Doubled: {}", doubler(5));

    // Safe division with Result
    match safe_divide(10, 2) {
        Ok(val) => println!("Division: {}", val),
        Err(e) => println!("Error: {}", e),
    }

    // Match expression
    println!("Classify 5: {}", classify_number(5));

    // Complex function
    println!("Complex: {}", complex_function(vec![1, -2, 3, -4, 5]));
}

/// Structure with methods containing blocks
struct DataProcessor {
    threshold: i32,
}

impl DataProcessor {
    fn new(threshold: i32) -> Self {
        DataProcessor { threshold }
    }

    /// Method with blocks
    fn process(&self, items: Vec<i32>) -> Vec<i32> {
        let mut result = Vec::new();

        // FOR with nested IF
        for item in items {
            if item > self.threshold {
                result.push(item);
            }
        }

        result
    }

    /// Method with match
    fn process_with_match(&self, value: i32) -> &'static str {
        match value {
            n if n < self.threshold => "below",
            n if n == self.threshold => "equal",
            _ => "above",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_data() {
        let result = process_data(vec![1, 10, 20], 10);
        assert_eq!(result, vec![1, 10, 40]);
    }

    #[test]
    fn test_closure() {
        let doubler = create_multiplier(2);
        assert_eq!(doubler(5), 10);
    }
}
