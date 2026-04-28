import numpy as np
import pandas as pd
from engine.models import LPProblem
from engine.standard import StandardSimplex

class TwoPhaseSimplex(StandardSimplex):
    def __init__(self, problem: LPProblem):
        # Initialize everything from the StandardSimplex parent class
        super().__init__(problem)
        self.artificial_cols = [] 

    def _build_phase_one_tableau(self) -> None:
        """Constructs Phase 1 tableau handling <=, >=, and = constraints."""
        num_constraints = len(self.problem.b)
        num_vars = len(self.problem.c)
        
        # Build extra columns in a single list to preserve interleaved ordering
        # so that column indices, variable names, and objective coefficients stay aligned.
        extra_cols = []
        var_names_extension = []
        phase1_obj_coeffs = []
        
        self.basic_vars = []
        self.artificial_cols = []
        current_col_idx = num_vars
        
        # Add extra columns (variables) for each constraint based on sign
        for i, c_type in enumerate(self.problem.types):
            if c_type == '<=':
                # Add Slack (+s)
                col = np.zeros(num_constraints)
                col[i] = 1
                extra_cols.append(col)
                var_names_extension.append(f"s{i+1}")
                phase1_obj_coeffs.append(0)
                
                self.basic_vars.append(current_col_idx)
                current_col_idx += 1
                
            elif c_type == '>=':
                # Add Surplus (-e) and Artificial (+a)
                col_surplus = np.zeros(num_constraints)
                col_surplus[i] = -1
                extra_cols.append(col_surplus)
                var_names_extension.append(f"e{i+1}")
                phase1_obj_coeffs.append(0)
                current_col_idx += 1
                
                col_art = np.zeros(num_constraints)
                col_art[i] = 1
                extra_cols.append(col_art)
                var_names_extension.append(f"a{i+1}")
                phase1_obj_coeffs.append(-1) # Phase 1 objective maximizes -sum(a)
                
                self.basic_vars.append(current_col_idx)
                self.artificial_cols.append(current_col_idx)
                current_col_idx += 1
                
            elif c_type == '=':
                # Add Artificial (+a)
                col_art = np.zeros(num_constraints)
                col_art[i] = 1
                extra_cols.append(col_art)
                var_names_extension.append(f"a{i+1}")
                phase1_obj_coeffs.append(-1)
                
                self.basic_vars.append(current_col_idx)
                self.artificial_cols.append(current_col_idx)
                current_col_idx += 1

        self.problem.var_names.extend(var_names_extension)
        
        # Assemble the matrices horizontally — single extra_cols list keeps column order consistent
        matrix_blocks = [self.problem.A]
        if extra_cols:
            matrix_blocks.append(np.column_stack(extra_cols))
            
        A_augmented = np.hstack((*matrix_blocks, self.problem.b.reshape(-1, 1)))
        
        # Build Phase 1 Z-row: Z* = -1 * artificials
        # Following StandardSimplex logic: Z-row = -C
        C_phase1 = np.array([0] * num_vars + phase1_obj_coeffs, dtype=float)
        obj_row = np.hstack((-C_phase1, [0.0]))
        
        self.tableau = np.vstack((A_augmented, obj_row))
        
        # INITIALIZE Z-ROW: Zero out artificial variables sitting in the basis
        for row_idx, basic_col in enumerate(self.basic_vars):
            if basic_col in self.artificial_cols:
                multiplier = self.tableau[-1, basic_col]
                self.tableau[-1, :] -= multiplier * self.tableau[row_idx, :]

    def _run_simplex_loop(self) -> str:
        while True:
            col = self._get_entering_variable()
            if col == -1:
                return 'optimal'
            row = self._get_leaving_variable(col)
            if row == -1:
                return 'unbounded'
            self._pivot(row, col)

    def _transition_to_phase_two(self) -> str:
        """Strips artificial columns, restores original objective, and re-initializes."""
        # 1. Feasibility Check: If Max Z* is less than 0, artificials are trapped in basis.
        if round(self.tableau[-1, -1], 5) < 0:
            return 'infeasible'
            
        # 2. Drop artificial columns from the matrix
        self.tableau = np.delete(self.tableau, self.artificial_cols, axis=1)
        
        # Update variable names list
        self.problem.var_names = [name for i, name in enumerate(self.problem.var_names) if i not in self.artificial_cols]
        
        # Update basic_vars tracking indices (since deleting columns shifted everything left)
        new_basic_vars = []
        for bv in self.basic_vars:
            shift = sum(1 for art_col in self.artificial_cols if art_col < bv)
            new_basic_vars.append(bv - shift)
        self.basic_vars = new_basic_vars
        
        # 3. Restore the original objective function
        num_current_vars = len(self.problem.var_names)
        num_original_vars = len(self.problem.c)
        c_original = np.zeros(num_current_vars)
        c_original[:num_original_vars] = self.problem.c # Only x vars get objective coeffs
        
        if self.problem.is_max:
            obj_row = np.hstack((-c_original, [0.0]))
        else:
            obj_row = np.hstack((c_original, [0.0]))
            
        self.tableau[-1, :] = obj_row
        
        # 4. Initialize Phase 2 Z-row (Zero out the basic variables)
        for row_idx, basic_col in enumerate(self.basic_vars):
            multiplier = self.tableau[-1, basic_col]
            if multiplier != 0:
                self.tableau[-1, :] -= multiplier * self.tableau[row_idx, :]
                
        return 'ready'

    def solve(self) -> dict:
        """Orchestrates Phase 1 and Phase 2 returning standard dictionary format."""
        # Inherited from StandardSimplex! Converts unrestricted into x+ and x-
        self._handle_unrestricted_variables()
        
        # === PHASE 1 ===
        self._build_phase_one_tableau()
        status = self._run_simplex_loop()
        
        if status == 'unbounded':
            return {'status': 'unbounded', 'solution': None, 'z': None}
            
        # === PHASE 2 ===
        transition_status = self._transition_to_phase_two()
        
        if transition_status == 'infeasible':
            return {'status': 'infeasible', 'solution': None, 'z': None}
            
        status = self._run_simplex_loop()
        
        if status == 'unbounded':
            return {'status': 'unbounded', 'solution': None, 'z': None}
            
        # Inherited from StandardSimplex! Recombines x+ and x-
        sol = self._extract_solution()
        z = self.tableau[-1, -1]
        
        if not self.problem.is_max:
            z = -z
            
        return {
            'status': 'optimal',
            'solution': sol,
            'z': z
        }