import logging
import threading
from typing import Any

import highspy
import numpy as np

from compass.globals import BETA, EXCHANGE_LIMIT
from compass.models.MetabolicModel import MetabolicModel
from .base import Optimizer, LinearProgramDelta, Solution

logger = logging.getLogger("compass")


def get_highs_config(threads: int | None = None, method: int | None = None,
                     overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Returns the HiGHS configuration parameters for Compass.
    These defaults are chosen for numerical stability and performance.

    Args:
        overrides: Optional dict of HiGHS option overrides using native
            HiGHS option names (e.g. output_flag, primal_feasibility_tolerance).
            Also supports the custom key 'solve_timeout'.
    """
    if threads is None:
        threads = 1
    if method is None:
        method = 0  # 0: Simplex (best for warm-start), 1: IPM
    config = {
        "output_flag": False,           # Disable all output
        "presolve": "off",              # Disable presolve (incompatible with warm-start basis)
        "threads": threads,             # Number of threads
        "solver": ["simplex", "ipm"][method],
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-12,
        # Per-solve timeout in seconds via cancelSolve callbacks.
        "solve_timeout": 30.0,
    }
    if overrides:
        config.update(overrides)
    return config


class HighsOptimizer(Optimizer):
    """
    Highs-based implementation of the Optimizer.

    Warm-start strategy: all possible variables (including synthetic exchange
    reactions) and constraints are pre-allocated at build time. Deltas are
    applied via in-place bound/cost changes only rather than adding/removing columns
    so HiGHS preserves the simplex basis across consecutive solves.
    """

    def __init__(self, model: MetabolicModel, config: dict[str, Any]):
        super().__init__(model)
        self.config = config
        self.__init_base_model()

    def __create_base_model(self):
        """
        Builds the initial Highs model from the provided metabolic model,
        including pre-allocated synthetic exchange variables (disabled via
        zero bounds).
        """
        problem = highspy.Highs()

        # Apply configuration (skip compass-specific keys that aren't HiGHS options)
        for key, value in self.config.items():
            if key in ("solve_timeout",):
                continue
            problem.setOptionValue(key, value)

        # Add reactions as variables
        variables = {}
        for id, reaction in self.model.reactions.items():
            lb = reaction.lower_bound
            ub = reaction.upper_bound
            var = problem.addVariable(lb=lb, ub=ub, name=reaction.id)
            variables[id] = var

        # Pre-allocate synthetic exchange variables.
        # These are added with bounds [0, 0] (disabled) and the correct
        # stoichiometric coefficient so they can be toggled via bound changes.
        synthetic = self.model.synthetic_exchange_reactions
        exchange_vars = {}

        # We'll collect (met_id, rxn_id, coeff, ub_when_active) for adding to constraints below
        pending_exchange = []

        for met_id, rxn_id in synthetic['secretion'].items():
            var = problem.addVariable(lb=0.0, ub=0.0, name=rxn_id)
            variables[rxn_id] = var
            exchange_vars[rxn_id] = var
            # Secretion: metabolite is consumed (coefficient -1 in mass balance)
            pending_exchange.append((met_id, rxn_id, -1.0))

        for met_id, rxn_id in synthetic['uptake'].items():
            var = problem.addVariable(lb=0.0, ub=0.0, name=rxn_id)
            variables[rxn_id] = var
            exchange_vars[rxn_id] = var
            # Uptake: metabolite is produced (coefficient +1 in mass balance)
            pending_exchange.append((met_id, rxn_id, 1.0))

        # Add metabolites as constraints
        constraints = {}
        for metab_id, stoichiometry in self.model.SMAT.items():
            # If there is no reaction associated with the given metabolite, then skip
            if len(stoichiometry) == 0:
                continue

            # x[0] is name of reaction
            # x[1] is stoichiometric coefficient of metabolite in reaction x[0]
            expr = sum([coeff * variables[id] for id, coeff in stoichiometry])

            # Each metabolite must obey mass conservation
            constraint = problem.addConstr(0 == expr, name=metab_id)
            constraints[metab_id] = constraint

        # Wire up pre-allocated exchange variables to their metabolite constraints
        for met_id, rxn_id, coeff in pending_exchange:
            if met_id in constraints:
                problem.chgCoeff(constraints[met_id], exchange_vars[rxn_id], coeff)

        return problem, variables, constraints, exchange_vars

    def __init_base_model(self):
        problem, variables, constraints, exchange_vars = self.__create_base_model()
        self.problem = problem
        self.variables = variables
        self.constraints = constraints
        self.exchange_vars = exchange_vars

        # Cache base bounds for all variables for efficient revert.
        # Indexed by column index for direct use with changeColBounds.
        lp = self.problem.getLp()
        n = self.problem.getNumCol()
        self._base_lb = list(lp.col_lower_[:n])
        self._base_ub = list(lp.col_upper_[:n])

    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Applies the delta via in-place bound/cost changes, solves the model,
        and reverts. No structural add/delete — basis is preserved for warm-start.
        """
        # Track which columns had bounds changed so we can revert only those
        changed_cols = []

        # Enable pre-allocated exchange variables by opening their bounds (B1).
        for met_id, rxn_id in delta.added_secretion.items():
            var = self.variables[rxn_id]
            col = int(var)
            changed_cols.append(col)
            self.problem.changeColBounds(col, 0.0, self.model.maximum_flux)

        for met_id, rxn_id in delta.added_uptake.items():
            var = self.variables[rxn_id]
            col = int(var)
            changed_cols.append(col)
            self.problem.changeColBounds(col, 0.0, EXCHANGE_LIMIT)

        # Block reactions by setting upper bound to lower bound
        for rxn_id in delta.blocked_reactions:
            var = self.variables[rxn_id]
            col = int(var)
            changed_cols.append(col)
            self.problem.changeColBounds(col, self._base_lb[col], self._base_lb[col])

        # High-flux constraints via lower bound change instead of adding constraint rows
        for rxn_id, limit in delta.high_flux.items():
            var = self.variables[rxn_id]
            col = int(var)
            changed_cols.append(col)
            self.problem.changeColBounds(col, limit, self._base_ub[col])

        # Set objective via batch cost change.
        n = self.problem.getNumCol()
        costs = np.zeros(n, dtype=np.float64)
        for rxn_id, coeff in delta.objective.items():
            if rxn_id in self.variables:
                costs[int(self.variables[rxn_id])] = coeff

        cols = np.arange(n, dtype=np.int32)
        self.problem.changeColsCost(n, cols, costs)

        if delta.sense == "max":
            self.problem.changeObjectiveSense(highspy.ObjSense.kMaximize)
        else:
            self.problem.changeObjectiveSense(highspy.ObjSense.kMinimize)

        self.solve_problem()

        model_status = self.problem.getModelStatus()
        success = model_status == highspy.HighsModelStatus.kOptimal

        if success:
            info = self.problem.getInfoValue("objective_function_value")
            obj_value = info[1]
        else:
            obj_value = None

        # Revert all bound changes
        for col in changed_cols:
            self.problem.changeColBounds(col, self._base_lb[col], self._base_ub[col])

        return Solution(success=success, status=str(model_status), obj_value=obj_value)

    ALL_SOLVER_METHODS = ["simplex", "ipm"]

    def solve_with_timeout(self):
        """Solves the current problem with a per-solve timeout using cancelSolve.

        Uses HandleUserInterrupt + threading.Timer to fire cancelSolve after
        the timeout. The interrupt callbacks (simplex/IPM) check a flag at
        iteration boundaries and stop the solver.

        Note: PDLP is excluded from ALL_SOLVER_METHODS because HiGHS has no
        PDLP interrupt callback, so cancelSolve cannot stop it.
        See @HandleUserInterrupt.setter in the highs's source code: 
        there are simplex, Ipm, and Mip callbacks, but no PDLP callback.
        """
        timeout = self.config.get("solve_timeout")
        if timeout is None:
            self.problem.solve()
            return

        # Reset poisoned flag from any previous cancelSolve call.
        # startSolve() does this but we call super().run() via solve(),
        # which skips startSolve() when HandleKeyboardInterrupt is False.
        self.problem._Highs__solver_should_stop = False

        self.problem.HandleUserInterrupt = True
        timer = threading.Timer(timeout, self.problem.cancelSolve)
        timer.start()
        try:
            self.problem.solve()
        finally:
            timer.cancel()
            self.problem.HandleUserInterrupt = False

    def solve_problem(self):
        """Solves the problem, retrying with alternate solver methods on non-optimal status."""
        self.solve_with_timeout()
        if self.problem.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            return

        configured_method = self.config.get("solver", "ipm")
        retry_methods = [m for m in self.ALL_SOLVER_METHODS if m != configured_method]

        for method in retry_methods:
            logger.info(
                "Received non-optimal status %s. Retrying with solver method '%s'",
                self.problem.getModelStatus(), method
            )
            try:
                self.problem.setOptionValue("solver", method)
                self.solve_with_timeout()
                if self.problem.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                    return
            finally:
                self.problem.setOptionValue("solver", configured_method)