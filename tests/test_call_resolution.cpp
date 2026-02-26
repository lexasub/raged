// C++ test file for enhanced call resolution testing
// Tests type inference, virtual method dispatch, lambda/closure extraction, and cross-file symbol resolution

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>

// Base class for virtual method dispatch testing
class Animal {
public:
    virtual void speak() const = 0;
    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    void speak() const override {
        std::cout << "Woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void speak() const override {
        std::cout << "Meow!" << std::endl;
    }
};

// Class for testing lambda/closure extraction
class MathOperations {
public:
    void testLambdas() {
        int x = 10;
        
        // Simple lambda
        auto add = [](int a, int b) { return a + b; };
        int result1 = add(5, 3);
        
        // Lambda with capture
        auto multiplyByX = [x](int a) { return a * x; };
        int result2 = multiplyByX(5);
        
        // Lambda with mutable capture
        int counter = 0;
        auto increment = [counter]() mutable { return ++counter; };
        int result3 = increment();
        
        // Lambda in container
        std::vector<std::function<int(int)>> operations;
        operations.push_back([](int a) { return a * 2; });
        operations.push_back([](int a) { return a + 10; });
        
        int result4 = operations[0](5);
        int result5 = operations[1](5);
    }
};

// Class for testing cross-file symbol resolution
class FileResolver {
public:
    void resolveSymbols() {
        // This will test cross-file resolution
        ExternalClass external;
        external.externalMethod();
        
        // Test with templates
        TemplateClass<int> intTemplate;
        intTemplate.doSomething();
        
        TemplateClass<std::string> stringTemplate;
        stringTemplate.doSomething();
    }
};

// External class defined in another file (for cross-file testing)
class ExternalClass {
public:
    void externalMethod() {
        std::cout << "External method called" << std::endl;
    }
    
    // Static method for testing
    static void staticExternalMethod() {
        std::cout << "Static external method called" << std::endl;
    }
    
    // Nested class for testing
    class NestedExternal {
    public:
        void nestedMethod() {
            std::cout << "Nested external method called" << std::endl;
        }
    };
};

// Template class for testing template instantiation
template <typename T>
class TemplateClass {
public:
    void doSomething() {
        T value = T();
        std::cout << "Template value: " << value << std::endl;
    }
    
    // Nested template class
    template <typename U>
    class NestedTemplate {
    public:
        void nestedDoSomething() {
            U value = U();
            std::cout << "Nested template value: " << value << std::endl;
        }
    };
};

// Function for testing function overloading
void overloadedFunction(int x) {
    std::cout << "Overloaded with int: " << x << std::endl;
}

void overloadedFunction(double x) {
    std::cout << "Overloaded with double: " << x << std::endl;
}

void overloadedFunction(const std::string& x) {
    std::cout << "Overloaded with string: " << x << std::endl;
}

// Function for testing variadic templates
template <typename... Args>
void variadicFunction(Args... args) {
    std::cout << "Variadic function called with " << sizeof...(args) << " arguments" << std::endl;
}

// Lambda factory function for testing
std::function<int(int)> createMultiplier(int factor) {
    return [factor](int x) { return x * factor; };
}

// Main function to test all features
int main() {
    // Test virtual method dispatch
    std::unique_ptr<Animal> dog = std::make_unique<Dog>();
    std::unique_ptr<Animal> cat = std::make_unique<Cat>();
    
    dog->speak();
    cat->speak();
    
    // Test lambda/closure extraction
    MathOperations mathOps;
    mathOps.testLambdas();
    
    // Test cross-file symbol resolution
    FileResolver resolver;
    resolver.resolveSymbols();
    
    // Test function overloading
    overloadedFunction(42);
    overloadedFunction(3.14);
    overloadedFunction("Hello");
    
    // Test variadic templates
    variadicFunction();
    variadicFunction(1);
    variadicFunction(1, 2.5, "test");
    
    // Test lambda factory
    auto multiplyBy3 = createMultiplier(3);
    int result = multiplyBy3(5);
    std::cout << "Lambda factory result: " << result << std::endl;
    
    return 0;
}