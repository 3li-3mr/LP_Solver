import numpy as np
from typing import List, Optional

class LPProblem:
    def __init__(self, objective_coeffs: List[float], constraint_matrix: List[List[float]], 
                 rhs_values: List[float], constraint_types: List[str], 
                 variable_restrictions: List[str], 
                 is_maximization: bool = True,
                 variable_names: Optional[List[str]] = None):
        
        self.c = np.array(objective_coeffs, dtype=float)
        self.A = np.array(constraint_matrix, dtype=float)
        self.b = np.array(rhs_values, dtype=float)
        self.types = constraint_types          # ['<=', '>=', '=']
        self.restrictions = variable_restrictions # ['>=0', 'unrestricted'] 
        self.is_max = is_maximization
        
        # Generate default labels (x1, x2, ..., xn)
        if variable_names is None:
            self.var_names = [f"x{i+1}" for i in range(len(objective_coeffs))]
        else:
            self.var_names = variable_names

        # Basic validation to prevent immediate engine crashes
        assert len(self.c) == len(self.restrictions), "Mismatch: objective coefficients and variable restrictions."
        assert len(self.A) == len(self.b) == len(self.types), "Mismatch: constraint matrix rows, RHS values, and types."