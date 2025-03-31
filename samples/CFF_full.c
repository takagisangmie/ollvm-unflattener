// Shout-out to Claude for generating this test file for me!

#include <stdio.h>
#include <stdlib.h>

// Helper function to calculate factorial
int calculate_factorial(int n)
{
    if (n <= 1)
        return 1;

    int result = 1;
    for (int i = 2; i <= n; i++)
    {
        result *= i;
    }
    return result;
}

// Function to calculate absolute value
int absolute_value(int n)
{
    return n < 0 ? -n : n;
}

// Helper function to compute powers
int compute_power(int base, int exponent)
{
    int result = 1;
    for (int i = 0; i < exponent; i++)
    {
        result *= base;
    }
    return result;
}

// Helper function to find greatest common divisor
int find_greatest_common_divisor(int a, int b)
{
    a = absolute_value(a);
    b = absolute_value(b);

    while (b != 0)
    {
        int temp = b;
        b = a % b;
        a = temp;
    }

    return a;
}

// Helper function to print binary representation
void print_binary(int num)
{
    if (num > 1)
    {
        print_binary(num / 2);
    }
    printf("%d", num % 2);
}

// The main target function
int target_function(int a)
{
    printf("Starting target_function with input: %d\n", a);

    // Variable declarations
    int result = 0;
    int temp = a;
    int floating_result = 0;

    // Basic arithmetic operations
    result = a * 5;
    printf("After multiplication by 5: %d\n", result);

    // Division with check for zero
    if (a != 0)
    {
        result = result / a;
        printf("After division by input: %d\n", result);
    }
    else
    {
        printf("Cannot divide by zero\n");
        result = 100; // Default value
    }

    // Conditional statement with multiple branches
    if (a > 100)
    {
        printf("Input is very large\n");
        result = result + 50;
    }
    else if (a > 50)
    {
        printf("Input is large\n");
        result = result + 25;
    }
    else if (a > 10)
    {
        printf("Input is medium\n");
        result = result + 10;
    }
    else
    {
        printf("Input is small\n");
        result = result + 5;
    }

    // Loop to perform some calculations
    printf("Starting loop calculations...\n");
    for (int i = 0; i < 5; i++)
    {
        temp += i * 2;
        printf("Loop iteration %d: temp = %d\n", i, temp);

        if (temp > 100)
        {
            printf("Breaking loop as temp exceeded 100\n");
            break;
        }
    }

    // Call to factorial function
    int factorial_result = calculate_factorial(absolute_value(a) % 10); // Use modulo to avoid large factorials
    printf("Factorial of %d is: %d\n", absolute_value(a) % 10, factorial_result);
    result += factorial_result;

    // Compute some powers
    floating_result = compute_power(a, 2);
    printf("Square of %d is: %d\n", a, floating_result);

    // Find GCD if applicable
    if (a != 0)
    {
        int gcd = find_greatest_common_divisor(a, 24);
        printf("GCD of %d and 24 is: %d\n", a, gcd);
        result += gcd;
    }

    // Another loop with different structure
    int j = 0;
    printf("Starting while loop...\n");
    while (j < absolute_value(a) % 5)
    {
        printf("While loop iteration %d\n", j);
        result += j;
        j++;
    }

    // Print the binary representation of our input
    printf("Binary representation of %d: ", a);
    print_binary(a);
    printf("\n");

    // Bit manipulation operations
    int bit_shifted = a << 2;
    printf("Value after left shift by 2: %d\n", bit_shifted);

    int bit_and = a & 0x0F;
    printf("Result of bitwise AND with 0x0F: %d\n", bit_and);

    // Final calculations
    result = result % 1000; // Keep result reasonable
    printf("Final result: %d\n", result);

    return result;
}

// Main function to demonstrate usage
int main()
{
    int test_values[] = {5, 25, 75, 150, 0};

    for (int i = 0; i < 5; i++)
    {
        printf("\n===== Testing target_function with input: %d =====\n", test_values[i]);
        int result = target_function(test_values[i]);
        printf("target_function returned: %d\n", result);
        printf("===============================================\n");
    }

    return 0;
}