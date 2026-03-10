"""
Python code example for demonstrating block extraction.

Contains:
- Functions with if/for/while blocks
- Lambdas with captured variables
- Try/except blocks
- With constructs
"""

from typing import Callable, List, Optional


def process_data(items: List[int], threshold: int = 10) -> List[int]:
    """Processes a list of numbers with various blocks."""
    result = []

    # IF block
    if not items:
        return []

    # FOR block with nested IF
    for item in items:
        if item > threshold:
            result.append(item * 2)
        elif item == threshold:
            result.append(item)
        # WHILE block inside FOR
        else:
            count = 0
            while count < 3:
                result.append(item + count)
                count += 1

    return result


def create_multiplier(factor: int) -> Callable[[int], int]:
    """Creates a lambda with a captured variable."""
    # Lambda with captured variable 'factor'
    return lambda x: x * factor


def safe_divide(a: int, b: int) -> Optional[float]:
    """Division with error handling."""
    # TRY block
    try:
        result = a / b
        # WITH block inside TRY
        with open("/tmp/result.txt", "w") as f:
            f.write(str(result))
        return result
    except ZeroDivisionError as e:
        print(f"Division by zero: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        print("Division attempt completed")


def complex_function(data: List[int]) -> int:
    """Function with multiple nested blocks."""
    total = 0

    # Level 1: FOR
    for i, value in enumerate(data):
        # Level 2: IF
        if value > 0:
            # Level 3: nested FOR
            for j in range(value):
                # Level 4: IF inside nested FOR
                if j % 2 == 0:
                    total += j
                else:
                    # Lambda inside function
                    transform = lambda x: x * 2 + i
                    total += transform(j)
        # Level 2: ELSE with WHILE
        else:
            count = abs(value)
            # Level 3: WHILE
            while count > 0:
                total -= 1
                count -= 1

    return total


# Global lambda
square = lambda x: x ** 2

# Lambda with multiple parameters
add = lambda x, y: x + y

# Lambda with capture from outer scope
multiplier = 10
scale = lambda x: x * multiplier


class DataProcessor:
    """Class with methods containing blocks."""

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        # Lambda in method
        self.check = lambda x: x > self.threshold

    def process(self, items: List[int]) -> List[int]:
        """Method with blocks."""
        result = []

        # FOR with nested IF
        for item in items:
            if self.check(item):
                result.append(item)

        return result

    def process_with_context(self, items: List[int]) -> List[int]:
        """Method with with context."""
        # WITH block
        with open("/tmp/data.txt", "w") as f:
            for item in items:
                if item > self.threshold:
                    f.write(str(item) + "\n")

        return items


if __name__ == "__main__":
    # Usage examples
    data = [1, 5, 10, 15, 20]

    # Call function with blocks
    result = process_data(data, threshold=10)
    print(f"Processed: {result}")

    # Lambda with captured variable
    doubler = create_multiplier(2)
    print(f"Doubled: {doubler(5)}")

    # Safe division with try/except
    print(f"Division: {safe_divide(10, 2)}")

    # Complex function
    print(f"Complex: {complex_function([1, -2, 3, -4, 5])}")
