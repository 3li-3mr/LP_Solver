import numpy as np
import pandas as pd
from engine.models import LPProblem

class StandardSimplex:
    def __init__(self, problem: LPProblem):
        self.problem = problem
        self.tableau = None
        self.basic_vars = None

    def _handle_unrestricted_variables(self) -> None:
        for i in range(len(self.problem.restrictions) - 1, -1, -1):
            if self.problem.restrictions[i].lower() == 'unrestricted':
                neg_c = -self.problem.c[i]
                self.problem.c = np.insert(self.problem.c, i + 1, neg_c)
                
                neg_col = -self.problem.A[:, i]
                self.problem.A = np.insert(self.problem.A, i + 1, neg_col, axis=1)
                
                base_name = self.problem.var_names[i]
                self.problem.var_names[i] = f"{base_name}+"
                self.problem.var_names.insert(i + 1, f"{base_name}-")

    def _build_initial_tableau(self) -> None:
        num_constraints = len(self.problem.b)
        num_vars = len(self.problem.c)
        slack_names = [f"s{i+1}" for i in range(num_constraints)]
        self.problem.var_names.extend(slack_names)

        slacks = np.eye(num_constraints)
        A_augmented = np.hstack((self.problem.A, slacks, self.problem.b.reshape(-1, 1)))

        if(self.problem.is_max == True):
            obj_row = np.hstack((-self.problem.c, np.zeros(num_constraints + 1)))
        else:
            obj_row = np.hstack((self.problem.c, np.zeros(num_constraints + 1)))
    
        self.tableau = np.vstack((A_augmented, obj_row))
        self.basic_vars = list(range(num_vars, num_vars + num_constraints))

    def _get_entering_variable(self) -> int:
        obj_row = self.tableau[-1, :-1]
        ent_col = int(np.argmin(obj_row))
        if(obj_row[ent_col] >= 0):
            return -1 #optimal
        return ent_col

    def _get_leaving_variable(self, pivot_col: int) -> int:
        col = self.tableau[:-1, pivot_col]
        b = self.tableau[:-1, -1]

        valid_rows = col > 0
        if not np.any(valid_rows):
            return -1 #unbound
        
        ratios = np.full_like(b, np.inf)
        ratios[valid_rows] = b[valid_rows] / col[valid_rows]
        return int(np.argmin(ratios))

    def _pivot(self, pivot_row: int, pivot_col: int) -> None:
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element

        for i in range(self.tableau.shape[0]):
            if(i != pivot_row):
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row, :]

        self.basic_vars[pivot_row] = pivot_col

    def _extract_iteration_state(self, step_num: int) -> pd.DataFrame:
        """Formats self.tableau into a labeled pandas DataFrame using self.basic_vars."""
        
        cols = self.problem.var_names + ['RHS']
        row_labels = [self.problem.var_names[var_idx] for var_idx in self.basic_vars]
        row_labels.append('Z')
        df = pd.DataFrame(self.tableau, columns=cols, index=row_labels)
        df.index.name = f"Iter {step_num}"
        
        return df
    
    def _extract_solution(self) -> dict:
        solution = {name: 0.0 for name in self.problem.var_names}

        for row, col in enumerate(self.basic_vars):
            var_name = self.problem.var_names[col]
            var_value = self.tableau[row, -1]
            solution[var_name] = var_value
            
        final_solution = {}
        for name in self.problem.var_names:
            if name.endswith('+'):
                base_name = name[:-1]
                final_solution[base_name] = solution[name] - solution[base_name + '-']
                
            elif name.endswith('-'):
                continue
                
            elif name.startswith('s'):
                continue
                
            else:
                final_solution[name] = solution[name]
                
        return final_solution

    def solve(self) -> dict:
        self._handle_unrestricted_variables()
        self._build_initial_tableau()
        history = []
        step_count = 0
        status = 'optimal'
        while(True):
            history.append(self._extract_iteration_state(step_count))
            step_count += 1
            col = self._get_entering_variable()
            if(col == -1):
                break #optimal
            row = self._get_leaving_variable(col)
            if(row == -1):
                status = 'unbounded'
                break 
            self._pivot(row, col)

        sol = self._extract_solution()
        z = self.tableau[-1, -1]
        if(self.problem.is_max == False):
            z = -z
        return {
            'status': status,
            'solution': sol,
            'z': z,
            'history': history
        }