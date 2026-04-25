import numpy as np
from typing import List

class LPProblem:
    def __init__(self, objective_coeffs: List[float], constraint_matrix: List[List[float]], 
                 rhs_values: List[float], constraint_types: List[str], 
                 is_maximization: bool = True):
        
        # Stored as separate arrays per your specification
        self.c = np.array(objective_coeffs, dtype=float)
        self.A = np.array(constraint_matrix, dtype=float)
        self.b = np.array(rhs_values, dtype=float)
        self.types = constraint_types # e.g., ['<=', '>=', '=']
        self.is_max = is_maximization