import unittest
import numpy as np
from scipy.optimize import linprog
from engine.models import LPProblem
from engine.standard import StandardSimplex
from engine.two_phase import TwoPhaseSimplex

class TestStandardSimplex(unittest.TestCase):

    def test_standard_maximization(self):
        """Validates a standard <= maximization problem."""
        # Maximize Z = 3x1 + 5x2
        # s.t. x1 <= 4; 2x2 <= 12; 3x1 + 2x2 <= 18
        c = [3.0, 5.0]
        A = [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]
        b = [4.0, 12.0, 18.0]
        restrictions = ['>=0', '>=0']
        
        problem = LPProblem(c, A, b, ['<=']*3, restrictions, is_maximization=True)
        solver = StandardSimplex(problem)
        result = solver.solve()

        # Oracle validation
        scipy_res = linprog([-3, -5], A_ub=A, b_ub=b, method='highs')

        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['z'], -scipy_res.fun)
        self.assertAlmostEqual(result['solution']['x1'], scipy_res.x[0])
        self.assertAlmostEqual(result['solution']['x2'], scipy_res.x[1])

    def test_standard_minimization(self):
        """Validates a standard <= minimization problem."""
        # Minimize Z = -2x1 - x2
        # s.t. x1 + x2 <= 5; x1 <= 3
        c = [-2.0, -1.0]
        A = [[1.0, 1.0], [1.0, 0.0]]
        b = [5.0, 3.0]
        restrictions = ['>=0', '>=0']
        
        problem = LPProblem(c, A, b, ['<=']*2, restrictions, is_maximization=False)
        solver = StandardSimplex(problem)
        result = solver.solve()

        # Oracle validation
        scipy_res = linprog(c, A_ub=A, b_ub=b, method='highs')

        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['z'], scipy_res.fun)
        self.assertAlmostEqual(result['solution']['x1'], scipy_res.x[0])
        self.assertAlmostEqual(result['solution']['x2'], scipy_res.x[1])

    def test_unbounded_problem(self):
        """Ensures the leaving variable check correctly flags an unbounded state."""
        # Maximize Z = 2x1 + x2
        # s.t. x1 - x2 <= 10
        c = [2.0, 1.0]
        A = [[1.0, -1.0]]
        b = [10.0]
        restrictions = ['>=0', '>=0']

        problem = LPProblem(c, A, b, ['<='], restrictions, is_maximization=True)
        solver = StandardSimplex(problem)
        result = solver.solve()

        self.assertEqual(result['status'], 'unbounded')

    def test_unrestricted_variables(self):
        """Validates xi+ and xi- recombination for unrestricted variables."""
        # Maximize Z = -x1
        # s.t. -x1 <= 5  =>  x1 >= -5
        # Therefore, max Z occurs when x1 = -5, yielding Z = 5.
        c = [-1.0]
        A = [[-1.0]]
        b = [5.0]
        restrictions = ['unrestricted']
        
        problem = LPProblem(c, A, b, ['<='], restrictions, is_maximization=True)
        solver = StandardSimplex(problem)
        result = solver.solve()

        # Oracle validation (bounds explicitly set to allow negative values)
        scipy_res = linprog([1.0], A_ub=A, b_ub=b, bounds=[(None, None)], method='highs')

        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['z'], -scipy_res.fun)
        self.assertAlmostEqual(result['solution']['x1'], scipy_res.x[0])

    def test_complex_mixed_restrictions(self):
        """
        Validates a multi-variable problem combining unrestricted and non-negative
        variables, designed to force unrestricted variables into negative optimal values.
        """
        # Maximize Z = -2x1 - 5x2 + 3x3
        # Subject to:
        #  x1 <= 2
        # -x1 <= 4  (Implicitly bounds x1 >= -4)
        #  x2 <= 3
        # -x2 <= 6  (Implicitly bounds x2 >= -6)
        #  x1 + x2 + x3 <= 10
        # -x1 - x2 + x3 <= 12
        # Restrictions: x1 unrestricted, x2 unrestricted, x3 >= 0
        
        c = [-2.0, -5.0, 3.0]
        A = [
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 1.0,  1.0,  1.0],
            [-1.0, -1.0,  1.0]
        ]
        b = [2.0, 4.0, 3.0, 6.0, 10.0, 12.0]
        restrictions = ['unrestricted', 'unrestricted', '>=0']
        
        problem = LPProblem(c, A, b, ['<='] * 6, restrictions, is_maximization=True)
        solver = StandardSimplex(problem)
        result = solver.solve()

        # Oracle validation
        # SciPy minimizes, so we negate the objective function coefficients.
        # We must explicitly define the bounds array for the oracle to map to our restrictions.
        bounds = [(None, None), (None, None), (0, None)]
        scipy_res = linprog([2.0, 5.0, -3.0], A_ub=A, b_ub=b, bounds=bounds, method='highs')

        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['z'], -scipy_res.fun)
        
        # Validate that the algebraic reconstruction correctly handles the negative values
        self.assertAlmostEqual(result['solution']['x1'], scipy_res.x[0])
        self.assertAlmostEqual(result['solution']['x2'], scipy_res.x[1])
        self.assertAlmostEqual(result['solution']['x3'], scipy_res.x[2])


class TestTwoPhaseSimplex(unittest.TestCase):

    def test_greater_than_constraints(self):
        """Validates a problem requiring Phase 1 surplus and artificial variables."""
        # Minimize Z = 2x1 + 3x2
        # s.t. x1 + x2 >= 4; 2x1 + x2 >= 5
        c = [2.0, 3.0]
        A = [[1.0, 1.0], [2.0, 1.0]]
        b = [4.0, 5.0]
        restrictions = ['>=0', '>=0']
        
        problem = LPProblem(c, A, b, ['>=', '>='], restrictions, is_maximization=False)
        solver = TwoPhaseSimplex(problem)
        result = solver.solve()

        # Oracle validation (SciPy requires <= for upper bounds, so we multiply by -1)
        scipy_res = linprog(c, A_ub=[[-1.0, -1.0], [-2.0, -1.0]], b_ub=[-4.0, -5.0], method='highs')

        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['z'], scipy_res.fun)
        self.assertAlmostEqual(result['solution']['x1'], scipy_res.x[0])
        self.assertAlmostEqual(result['solution']['x2'], scipy_res.x[1])

    def test_equality_constraints(self):
        """Validates a problem with equality constraints (artificial variables only)."""
        # Maximize Z = 3x1 + x2
        # s.t. x1 + x2 = 3; x1 <= 2
        c = [3.0, 1.0]
        A = [[1.0, 1.0], [1.0, 0.0]]
        b = [3.0, 2.0]
        restrictions = ['>=0', '>=0']
        
        problem = LPProblem(c, A, b, ['=', '<='], restrictions, is_maximization=True)
        solver = TwoPhaseSimplex(problem)
        result = solver.solve()

        # Oracle validation (SciPy minimizes by default, so negate objective)
        scipy_res = linprog([-3.0, -1.0], A_eq=[[1.0, 1.0]], b_eq=[3.0], A_ub=[[1.0, 0.0]], b_ub=[2.0], method='highs')

        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['z'], -scipy_res.fun)
        self.assertAlmostEqual(result['solution']['x1'], scipy_res.x[0])
        self.assertAlmostEqual(result['solution']['x2'], scipy_res.x[1])

    def test_infeasible_problem(self):
        """Ensures Phase 1 correctly identifies an infeasible problem (artificials > 0)."""
        # Maximize Z = x1 + x2
        # s.t. x1 + x2 <= 2; x1 + x2 >= 4  (Logically impossible)
        c = [1.0, 1.0]
        A = [[1.0, 1.0], [1.0, 1.0]]
        b = [2.0, 4.0]
        restrictions = ['>=0', '>=0']
        
        problem = LPProblem(c, A, b, ['<=', '>='], restrictions, is_maximization=True)
        solver = TwoPhaseSimplex(problem)
        result = solver.solve()

        # If Phase 1 finishes and Max Z* < 0, the solver should return 'infeasible'
        self.assertEqual(result['status'], 'infeasible')
        self.assertIsNone(result['solution'])



if __name__ == '__main__':
    unittest.main()