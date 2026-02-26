// Header file for cross-file symbol resolution testing
// This file will be included in test_call_resolution.cpp to test cross-file resolution

#ifndef TEST_CALL_RESOLUTION_H
#define TEST_CALL_RESOLUTION_H

#include <iostream>
#include <string>
#include <vector>

// External class for cross-file testing
class ExternalClass {
public:
    void externalMethod();
    
    // Static method for testing
    static void staticExternalMethod();
    
    // Nested class for testing
    class NestedExternal {
    public:
        void nestedMethod();
    };
};

// Template class for testing template instantiation
template <typename T>
class TemplateClass {
public:
    void doSomething();
    
    // Nested template class
    template <typename U>
    class NestedTemplate {
    public:
        void nestedDoSomething();
    };
};

// Function overloading for testing
void overloadedFunction(int x);
void overloadedFunction(double x);
void overloadedFunction(const std::string& x);

// Variadic template function for testing
template <typename... Args>
void variadicFunction(Args... args);

// Lambda factory function for testing
std::function<int(int)> createMultiplier(int factor);

#endif // TEST_CALL_RESOLUTION_H