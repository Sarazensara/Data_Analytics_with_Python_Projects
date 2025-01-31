"""
Tests for GUI.py
"""
import unittest
import tkinter as tk
import numpy as np
from GUI import CalculatorStack, MatrixCalculator

class TestCalculatorStack(unittest.TestCase):
    """Test methods for the CalculatorStack class."""

    def test_push_pop(self):
        """Test the push and pop methods."""
        stack = CalculatorStack()
        stack.push((2, 3))
        self.assertEqual(stack.pop(), (2, 3))

    def test_size(self):
        """Test the size method."""
        stack = CalculatorStack()
        stack.push((1, 1))
        stack.push((2, 2))
        self.assertEqual(stack.size(), 2)

    def test_multiply(self):
        """Test the multiply method."""
        stack = CalculatorStack()
        stack.push((2, 3))
        stack.multiply()
        self.assertEqual(stack.pop(), 6)

class TestMatrixCalculator(unittest.TestCase):
    """Test methods for the MatrixCalculator class."""

    def assert_matrices_equal(self, result, expected_result):
        np.testing.assert_allclose(result, expected_result, atol=1e-8)

    def test_add_matrices(self):
        """Test the add_matrices method."""
        calculator = MatrixCalculator()
        calculator.matrix_a_display = tk.Text(calculator.window)
        calculator.matrix_b_display = tk.Text(calculator.window)
        calculator.result_display = tk.Text(calculator.window)
        calculator.matrix_a_display.insert('1.0', '[[1,2],[3,4]]')
        calculator.matrix_b_display.insert('1.0', '[[5,6],[7,8]]')
        calculator.add_matrices()
        result = calculator.result_display.get('1.0', 'end-1c')
        expected_result = '[[6,8],[10,12]]'
        self.assertEqual(result, expected_result)

    def test_multiply_matrices(self):
        """Test the multiply_matrices method."""
        calculator = MatrixCalculator()
        calculator.matrix_a_display = tk.Text(calculator.window)
        calculator.matrix_b_display = tk.Text(calculator.window)
        calculator.result_display = tk.Text(calculator.window)
        calculator.matrix_a_display.insert('1.0', '[[1,2],[3,4]]')
        calculator.matrix_b_display.insert('1.0', '[[5,6],[7,8]]')
        calculator.multiply_matrices(element_wise=False)
        result = calculator.result_display.get('1.0', 'end-1c')
        expected_result = '[[19,22],[43,50]]'
        self.assertEqual(result, expected_result)

    def test_subtract_matrices(self):
        """Test the subtract_matrices method."""
        calculator = MatrixCalculator()
        calculator.matrix_a_display = tk.Text(calculator.window)
        calculator.matrix_b_display = tk.Text(calculator.window)
        calculator.result_display = tk.Text(calculator.window)
        calculator.matrix_a_display.insert('1.0', '[[1,2],[3,4]]')
        calculator.matrix_b_display.insert('1.0', '[[5,6],[7,8]]')
        calculator.subtract_matrices()
        result = calculator.result_display.get('1.0', 'end-1c')
        expected_result = '[[-4,-4],[-4,-4]]'
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()