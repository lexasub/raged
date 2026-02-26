// Java test file for enhanced call resolution testing
// Tests type inference, virtual method dispatch, lambda/closure extraction, and cross-file symbol resolution

import java.util.*;
import java.util.function.*;

// Base class for virtual method dispatch testing
abstract class Animal {
    public abstract void speak();
}

class Dog extends Animal {
    @Override
    public void speak() {
        System.out.println("Woof!");
    }
}

class Cat extends Animal {
    @Override
    public void speak() {
        System.out.println("Meow!");
    }
}

// Interface for testing lambda/closure extraction
interface MathOperation {
    int calculate(int a, int b);
}

class MathOperations {
    public void testLambdas() {
        int x = 10;
        
        // Simple lambda
        MathOperation add = (a, b) -> a + b;
        int result1 = add.calculate(5, 3);
        
        // Lambda with capture
        MathOperation multiplyByX = (a) -> a * x;
        int result2 = multiplyByX.calculate(5);
        
        // Lambda with mutable capture (using array trick)
        final int[] counter = {0};
        IntSupplier increment = () -> ++counter[0];
        int result3 = increment.getAsInt();
        
        // Lambda in collection
        List<MathOperation> operations = new ArrayList<>();
        operations.add((a) -> a * 2);
        operations.add((a) -> a + 10);
        
        int result4 = operations.get(0).calculate(5);
        int result5 = operations.get(1).calculate(5);
    }
}

// Class for testing cross-file symbol resolution
class FileResolver {
    public void resolveSymbols() {
        // This will test cross-file resolution
        ExternalClass external = new ExternalClass();
        external.externalMethod();
        
        // Test with generics
        TemplateClass> intTemplate = new TemplateClass>();
        intTemplate.doSomething();
        
        TemplateClass> stringTemplate = new TemplateClass>();
        stringTemplate.doSomething();
    }
}

// External class defined in another file (for cross-file testing)
class ExternalClass {
    public void externalMethod() {
        System.out.println("External method called");
    }
    
    // Static method for testing
    public static void staticExternalMethod() {
        System.out.println("Static external method called");
    }
    
    // Nested class for testing
    public static class NestedExternal {
        public void nestedMethod() {
            System.out.println("Nested external method called");
        }
    }
}

// Generic class for testing generic instantiation
class TemplateClass<T> {
    public void doSomething() {
        T value = null;
        System.out.println("Template value: " + value);
    }
    
    // Nested generic class
    public static class NestedTemplate<U> {
        public void nestedDoSomething() {
            U value = null;
            System.out.println("Nested template value: " + value);
        }
    }
}

// Interface for testing method overloading
interface OverloadedInterface {
    void overloadedMethod(int x);
    void overloadedMethod(double x);
    void overloadedMethod(String x);
}

// Class for testing method overloading
class OverloadedClass implements OverloadedInterface {
    @Override
    public void overloadedMethod(int x) {
        System.out.println("Overloaded with int: " + x);
    }
    
    @Override
    public void overloadedMethod(double x) {
        System.out.println("Overloaded with double: " + x);
    }
    
    @Override
    public void overloadedMethod(String x) {
        System.out.println("Overloaded with string: " + x);
    }
}

// Interface for testing functional programming
@FunctionalInterface
interface FunctionFactory {
    IntUnaryOperator create(int factor);
}

// Main class to test all features
public class TestCallResolution {
    public static void main(String[] args) {
        // Test virtual method dispatch
        Animal dog = new Dog();
        Animal cat = new Cat();
        
        dog.speak();
        cat.speak();
        
        // Test lambda/closure extraction
        MathOperations mathOps = new MathOperations();
        mathOps.testLambdas();
        
        // Test cross-file symbol resolution
        FileResolver resolver = new FileResolver();
        resolver.resolveSymbols();
        
        // Test method overloading
        OverloadedClass overloaded = new OverloadedClass();
        overloaded.overloadedMethod(42);
        overloaded.overloadedMethod(3.14);
        overloaded.overloadedMethod("Hello");
        
        // Test functional programming
        FunctionFactory factory = (factor) -> (x) -> x * factor;
        IntUnaryOperator multiplyBy3 = factory.create(3);
        int result = multiplyBy3.applyAsInt(5);
        System.out.println("Lambda factory result: " + result);
        
        // Test streams and method references
        List> numbers = Arrays.asList(1, 2, 3, 4, 5);
        int sum = numbers.stream()
                         .mapToInt(Integer::intValue)
                         .sum();
        System.out.println("Stream sum: " + sum);
        
        // Test default methods in interfaces
        InterfaceWithDefaultMethods defaultMethods = new InterfaceWithDefaultMethods() {};
        defaultMethods.defaultMethod();
        defaultMethods.anotherDefaultMethod();
    }
}

// Interface with default methods for testing
interface InterfaceWithDefaultMethods {
    default void defaultMethod() {
        System.out.println("Default method called");
    }
    
    default void anotherDefaultMethod() {
        System.out.println("Another default method called");
    }
}