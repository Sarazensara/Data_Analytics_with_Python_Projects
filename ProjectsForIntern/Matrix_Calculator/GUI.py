import tkinter as tk
import numpy as np
import math
'''
#  File: GUI.py

#  Description: Generates a scalar and matrix calculator upon user input.

#  Data Structures featured: Stacks, Lists, Dictionaries

#  Algorithms featured: Matrix calculations

#  Student Name: Odette Saenz

#  Student UT EID: oss286

#  Partner Name: Joshua Garcia

#  Partner UT EID: jcg4725

#  Course Name: CS 313E

#  Unique Number:50775

#  Date Created:4/19/2024

#  Date Last Modified:4/22/2024
'''
class CalculatorStack:
    """A stack data structure for a calculator."""
    def __init__(self):
        """Initialize an empty stack."""
        self.stack = []

    def push(self, item):
        """Push an item onto the stack."""
        self.stack.append(item)

    def pop(self):
        """Pop an item from the stack."""
        if not self.is_empty():
            return self.stack.pop()
        else:
            # Return a default value when the stack is empty
            return ()

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.stack) == 0

    def size(self):
        """Return the size of the stack."""
        return len(self.stack)

    def multiply(self):
        """Multiply the top two items on the stack."""
        operand1, operand2 = self.pop()
        result = operand1 * operand2
        self.stack.append(result)

    def divide(self):
        """Divide the top two items on the stack."""
        operand1, operand2 = self.pop()
        try:
            result = round(operand2 / operand1, 2)
        except ZeroDivisionError:
            print("You can't divide by 0!")
        else:
            self.stack.append(result)

    def add(self):
        """Add the top two items on the stack."""
        operand1, operand2 = self.pop()
        result = operand1 + operand2
        self.stack.append(result)

    def subtract(self):
        """Subtract the top two items on the stack."""
        operand1, operand2 = self.pop()
        result = operand2 - operand1
        self.stack.append(result)

    def exponent(self):
        """Calculate the exponentiation of the top two items on the stack."""
        operand1, operand2 = self.pop()
        result = operand2 ** operand1
        self.stack.append(result)

    def log(self):
        """Calculate the logarithm of the top item on the stack."""
        operand1, operand2 = self.pop()
        result = math.log(operand2, operand1)
        self.stack.append(result)

    def natural_log(self):
        """Calculate the natural logarithm of the top item on the stack."""
        operand = self.pop()
        result = math.log(operand)
        self.stack.append(result)

    def sine(self):
        """Calculate the sine of the top item on the stack (in degrees)."""
        operand = self.pop()
        result = math.sin(math.radians(operand))
        self.stack.append(result)

    def cosine(self):
        """Calculate the cosine of the top item on the stack (in degrees)."""
        operand = self.pop()
        result = math.cos(math.radians(operand))
        self.stack.append(result)

    def tangent(self):
        """Calculate the tangent of the top item on the stack (in degrees)."""
        operand = self.pop()
        result = math.tan(math.radians(operand))
        self.stack.append(result)

    def absolute(self):
        """Calculate the absolute value of the top item on the stack."""
        operand = self.pop()
        result = abs(operand)
        self.stack.append(result)

class MatrixCalculator:
    """A matrix calculator GUI using Tkinter."""

    def __init__(self):
        """Initialize the matrix calculator."""
        # Initialize the Tkinter window
        self.window = tk.Tk()
        self.window.title('Matrix Calculator')
        self.window.geometry('800x600')  # Set the size to 800x600 pixels

        # Initialize the matrix calculator widgets
        self.mode_label = tk.Label(self.window, text="Select Calculation Mode:")
        self.mode_label.pack()

        self.scalar_button = tk.Button(self.window, text="Scalar", command=self.set_scalar_mode)
        self.scalar_button.pack()

        self.matrix_button = tk.Button(self.window, text="Matrix", command=self.set_matrix_mode)
        self.matrix_button.pack()

        self.matrix_a_display = None
        self.matrix_b_display = None
        self.result_display = None

        self.matrix_stack = CalculatorStack()

        self.operator_functions = {
            '+': self.matrix_stack.add,
            '-': self.matrix_stack.subtract,
            '*': self.matrix_stack.multiply,
            '/': self.matrix_stack.divide,
            '^': self.matrix_stack.exponent,
            'log': self.matrix_stack.log,
            'ln': self.matrix_stack.natural_log,
            'sin': self.matrix_stack.sine,
            'cos': self.matrix_stack.cosine,
            'tan': self.matrix_stack.tangent,
            'abs': self.matrix_stack.absolute
        }

    def set_scalar_mode(self):
        """Switch the calculator to scalar mode."""
        self.clear_widgets()

        self.create_scalar_widgets()

    def set_matrix_mode(self):
        """Switch the calculator to matrix mode."""
        self.clear_widgets()

        self.create_matrix_widgets()

    def create_scalar_widgets(self):
        """Configure widgets for the scalar calculator."""
        # Create and configure the button widgets for scalar calculator
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=20)  # Added padding to avoid any overlap or layout issues

        # Create a text widget for displaying numbers
        self.display = tk.Entry(button_frame, font=('Arial', 24), width=15, borderwidth=5, justify='right')
        self.display.grid(row=0, column=0, columnspan=4, padx=10, pady=10)  # Ensure the display is placed properly

        # Create number buttons (1-9)
        for i in range(1, 10):
            row = 3 - (i-1) // 3
            col = (i-1) % 3
            button = tk.Button(button_frame, text=str(i), font=('Arial', 18),
                            command=lambda i=i: self.add_to_display(str(i)))
            button.grid(row=row, column=col, padx=10, pady=10)

        # Button for '0'
        zero_button = tk.Button(button_frame, text='0', font=('Arial', 18),
                                command=lambda: self.add_to_display('0'))
        zero_button.grid(row=4, column=1, padx=10, pady=10)

        # Operation buttons (+, -, *, /, ^)
        operations = ['+', '-', '*', '/', '^']
        for i, op in enumerate(operations):
            button = tk.Button(button_frame, text=op, font=('Arial', 18),
                            command=lambda op=op: self.add_to_display(op))
            button.grid(row=i+1, column=3, padx=10, pady=10)

        # Function buttons (log, abs, ln, sin, cos, tan)
        functions = ['log', 'abs', 'ln', 'sin', 'cos', 'tan']
        for i, func in enumerate(functions):
            button = tk.Button(button_frame, text=func, font=('Arial', 18),
                            command=lambda func=func: self.add_to_display(func + '('))
            button.grid(row=i+1, column=4, padx=10, pady=10)

        # Button for ')'
        complete_button = tk.Button(button_frame, text=')', font=('Arial', 18),
                                    command=lambda: self.add_to_display(')'))
        complete_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # Clear and Equals buttons
        clear_button = tk.Button(button_frame, text='C', font=('Arial', 18),
                                command=self.clear_display)
        clear_button.grid(row=5, column=2, columnspan=2, padx=10, pady=10)

        equals_button = tk.Button(button_frame, text='=', font=('Arial', 18),
                                command=self.calculate)
        equals_button.grid(row=5, column=3, columnspan=2, padx=10, pady=10)

    def add_to_display(self, value):
        current_value = self.display.get()
        self.display.delete(0, tk.END)
        self.display.insert(tk.END, current_value + value)

    def clear_display(self):
        """Clear the display field."""
        self.display.delete(0, tk.END)

    def calculate(self):
        """Calculate the expression in the display field."""
        expression = self.display.get()
        try:
            # Use eval to evaluate the expression entered
            result = eval(expression)
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        except Exception as e:
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, "Error")

    def create_matrix_widgets(self):
        """Creates the matrix display screen and buttons for the matrix calculator."""
        matrix_frame = tk.Frame(self.window)
        matrix_frame.pack(expand=True, fill='both')

        # Header
        header_label = tk.Label(matrix_frame, text="Input matrix in this format: [[1,2,3],[2,2,2]]", font=('Arial', 14), pady=10)
        header_label.pack()

        matrix_label_a = tk.Label(matrix_frame, text="Matrix A:")
        matrix_label_a.pack()

        self.matrix_a_display = tk.Text(matrix_frame, height=5, width=20, font=('Arial', 12))
        self.matrix_a_display.pack()

        matrix_label_b = tk.Label(matrix_frame, text="Matrix B:")
        matrix_label_b.pack()

        self.matrix_b_display = tk.Text(matrix_frame, height=5, width=20, font=('Arial', 12))
        self.matrix_b_display.pack()

        result_label = tk.Label(matrix_frame, text="Result:")
        result_label.pack()

        self.result_display = tk.Text(matrix_frame, height=5, width=20, font=('Arial', 12))
        self.result_display.pack()

        # Button frame
        button_frame = tk.Frame(matrix_frame)
        button_frame.pack()

        # Buttons for matrix operations
        add_button = tk.Button(button_frame, text='Add', font=('Arial', 18), padx=20, pady=10,
                            command=self.add_matrices)
        add_button.pack(side=tk.LEFT, padx=10, pady=10)

        subtract_button = tk.Button(button_frame, text='Subtract', font=('Arial', 18), padx=20, pady=10,
                                    command=self.subtract_matrices)
        subtract_button.pack(side=tk.LEFT, padx=10, pady=10)

        multiply_button = tk.Button(button_frame, text='Multiply', font=('Arial', 18), padx=20, pady=10,
                                    command=lambda: self.multiply_matrices(element_wise=False))
        multiply_button.pack(side=tk.LEFT, padx=10, pady=10)

        element_wise_multiply_button = tk.Button(button_frame, text='Element-wise Multiply', font=('Arial', 18), padx=20, pady=10,
                                                command=lambda: self.multiply_matrices(element_wise=True))
        element_wise_multiply_button.pack(side=tk.LEFT, padx=10, pady=10)

        divide_button = tk.Button(button_frame, text='Divide', font=('Arial', 18), padx=20, pady=10,
                                command=self.divide_matrices)
        divide_button.pack(side=tk.LEFT, padx=10, pady=10)

    def add_matrices(self):
        """Adds the matrices displayed in the matrix display to perform matrix addition."""
        matrix_a_str = self.matrix_a_display.get('1.0', tk.END)
        matrix_b_str = self.matrix_b_display.get('1.0', tk.END)
        try:
            matrix_a = np.array(eval(matrix_a_str))
            matrix_b = np.array(eval(matrix_b_str))
            self.matrix_stack.push(matrix_a)
            self.matrix_stack.push(matrix_b)
            result = self.matrix_stack.pop() + self.matrix_stack.pop()
            self.result_display.delete('1.0', tk.END)
            self.result_display.insert(tk.END, str(result))
        except Exception as e:
            self.error_label.config(text="Error: Please enter valid matrices in the correct format.")


    def create_matrix_display(self):
        """Creates a display widget for a matrix."""
        # Creates display widget
        matrix_frame = tk.Frame(self.window)
        matrix_frame.pack(expand=True, fill='both')

        matrix_label = tk.Label(matrix_frame, text="Matrix:")
        matrix_label.pack()

        matrix_display = tk.Text(matrix_frame, height=5, width=20, font=('Arial', 12))
        matrix_display.pack()

        return matrix_display

    def subtract_matrices(self):
        """Subtracts the matrices displayed in the matrix display."""
        matrix_a_str = self.matrix_a_display.get('1.0', tk.END)
        matrix_b_str = self.matrix_b_display.get('1.0', tk.END)
        try:
            matrix_a = np.array(eval(matrix_a_str))
            matrix_b = np.array(eval(matrix_b_str))
            self.matrix_stack.push(matrix_a)
            self.matrix_stack.push(matrix_b)
            result = self.matrix_stack.pop() - self.matrix_stack.pop()  # Subtract in the correct order
            self.result_display.delete('1.0', tk.END)
            self.result_display.insert(tk.END, str(result))
        except Exception as e:
            print(e)

    def multiply_matrices(self, element_wise=False):
        """Multiplies the matrices displayed in the matrix display.

        Args:
            element_wise (bool): Whether to perform element-wise multiplication or matrix multiplication.
        """
        matrix_a_str = self.matrix_a_display.get('1.0', tk.END)
        matrix_b_str = self.matrix_b_display.get('1.0', tk.END)
        try:
            matrix_a = np.array(eval(matrix_a_str))
            matrix_b = np.array(eval(matrix_b_str))
            self.matrix_stack.push(matrix_a)
            self.matrix_stack.push(matrix_b)
            if element_wise:
                result = np.multiply(matrix_a, matrix_b)
            else:
                result = np.matmul(matrix_a, matrix_b)
            self.result_display.delete('1.0', tk.END)
            self.result_display.insert(tk.END, str(result))
        except Exception as e:
            print(e)

    def divide_matrices(self):
        """Divides the matrices displayed in the matrix display."""
        matrix_a_str = self.matrix_a_display.get('1.0', tk.END)
        matrix_b_str = self.matrix_b_display.get('1.0', tk.END)
        try:
            matrix_a = np.array(eval(matrix_a_str))
            matrix_b = np.array(eval(matrix_b_str))
            self.matrix_stack.push(matrix_a)
            self.matrix_stack.push(matrix_b)
            operand2 = self.matrix_stack.pop()
            operand1 = self.matrix_stack.pop()
            result = np.divide(operand1, operand2)
            self.result_display.delete('1.0', tk.END)
            self.result_display.insert(tk.END, str(result))
        except Exception as e:
            print(e)

    def clear_widgets(self):
        """Clears all widgets from the window."""
        for widget in self.window.winfo_children():
            widget.destroy()

    def add_to_display(self, value):
        """Adds a value to the display widget.

        Args:
            value (str): The value to add to the display.
        """
        current_value = self.display.get()
        if value == 'abs(':
            new_value = current_value + 'abs('
        else:
            new_value = current_value + value
        self.display.delete(0, tk.END)
        self.display.insert(0, new_value)

    def clear_display(self):
        """Clear the display widget."""
        self.display.delete(0, tk.END)

    def calculate(self):
        """Evaluate the expression in the display and update the display with the result."""
        expression = self.display.get()
        try:
            # Replace '^' with '**' for exponentiation
            expression = expression.replace('^', '**')

            # Evaluate the modified expression
            result = eval(expression)

            # Update the display with the result
            self.display.delete(0, tk.END)
            self.display.insert(0, str(result))
        except Exception as e:
            # Handle specific math function calculations
            try:
                # Extract the function and argument
                function, arg_start, arg_end = self.extract_function(expression)

                # Calculate the function value
                result = self.calculate_function(function, expression, arg_start, arg_end)

                # Replace the function expression with the result
                expression = expression.replace(expression[arg_start:arg_end + 1], str(result))

                # Evaluate the modified expression
                result = eval(expression)
                self.display.delete(0, tk.END)
                self.display.insert(0, str(result))
            except Exception as e:
                # Display error message if calculation fails
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")

    def open(self):
        """Start the main event loop for the calculator window."""
        self.window.mainloop()

"""
Without numpy:

def add_matrices(matrix_a, matrix_b):
    "Add two matrices."
    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            row.append(matrix_a[i][j] + matrix_b[i][j])
        result.append(row)
    return result

def subtract_matrices(matrix_a, matrix_b):
    "Subtract one matrix from another."
    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            row.append(matrix_a[i][j] - matrix_b[i][j])
        result.append(row)
    return result

def multiply_matrices(matrix_a, matrix_b, element_wise=False):
    "Multiply two matrices."
    if element_wise:
        result = []
        for i in range(len(matrix_a)):
            row = []
            for j in range(len(matrix_a[0])):
                row.append(matrix_a[i][j] * matrix_b[i][j])
            result.append(row)
        return result
    else:
        # Matrix multiplication
        result = []
        for i in range(len(matrix_a)):
            row = []
            for j in range(len(matrix_b[0])):
                val = 0
                for k in range(len(matrix_a[0])):
                    val += matrix_a[i][k] * matrix_b[k][j]
                row.append(val)
            result.append(row)
        return result

def divide_matrices(matrix_a, matrix_b):
    "Divide one matrix by another."
    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            if matrix_b[i][j] != 0:
                row.append(matrix_a[i][j] / matrix_b[i][j])
            else:
                row.append(float('inf'))  # Handle division by zero
        result.append(row)
    return result
"""    

calculator = MatrixCalculator()

if __name__ == '__main__':
    calculator.open()