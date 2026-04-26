import logging
import threading
from typing import Any

import highspy

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
        method = 1  # 0: Simplex, 1: IPM
    config = {
        "output_flag": False,           # Disable all output
        "presolve": "on",               # Enable presolve
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
    """

    def __init__(self, model: MetabolicModel, config: dict[str, Any]):
        super().__init__(model)
        self.config = config
        self.__init_base_model()

    def __create_base_model(self):
        """
        Builds the initial Highs model from the provided metabolic model.
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

        return problem, variables, constraints
    
    def __init_base_model(self):
        problem, variables, constraints = self.__create_base_model()
        self.problem = problem
        self.variables = variables
        self.constraints = constraints

    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Applies the delta, solves the model, and returns the solution.
        Reverts changes to the model after solving, to serve as tabula rasa for the next delta.
        """
        # Store original state to revert later
        # With highs, we should always be able to revert the delta
        original_bounds = {}
        added_vars = []
        added_constraints = []

        for met_id, rxn_id in delta.added_secretion.items():
            met_constr = self.constraints[met_id]
            rxn_var = self.problem.addVariable(lb=0.0, ub=self.model.maximum_flux, name=rxn_id)
            added_vars.append(rxn_var)
            # Add secretion to metabolite's constraint as a reduction in metabolite
            self.problem.chgCoeff(met_constr, rxn_var, -1.0)

        for met_id, rxn_id in delta.added_uptake.items():
            met_constr = self.constraints[met_id]
            rxn_var = self.problem.addVariable(lb=0.0, ub=EXCHANGE_LIMIT, name=rxn_id)
            added_vars.append(rxn_var)
            # Add uptake to metabolite's constraint as an increase in metabolite
            self.problem.chgCoeff(met_constr, rxn_var, 1.0)

        # Close all blocked reactions by setting upper bound to lower bound
        # and store previous bounds so they can be restored later
        lp = self.problem.getLp()
        for rxn_id in delta.blocked_reactions:
            var = self.variables[rxn_id]
            old_lb = lp.col_lower_[int(var)]
            old_ub = lp.col_upper_[int(var)]
            original_bounds[rxn_id] = (old_lb, old_ub)
            self.problem.changeColBounds(int(var), old_lb, old_lb)

        for rxn_id, limit in delta.high_flux.items():
            var = self.variables[rxn_id]
            added_constraints.append(self.problem.addConstr(var >= limit, name=f"{rxn_id}_REACTION_OPT"))

        # Construct objective linear expression by summing all coefficient * reaction pairs.
        obj_expr = sum([coeff * self.variables[id] for id, coeff in delta.objective.items()])

        if delta.sense == "max":
            sense = highspy.ObjSense.kMaximize
        else:
            sense = highspy.ObjSense.kMinimize
        self.problem.setObjective(obj_expr, sense)

        self.solve_problem()

        model_status = self.problem.getModelStatus()
        if model_status == highspy.HighsModelStatus.kOptimal:
            success = True
        else:
            success = False

        if success:
            info = self.problem.getInfoValue("objective_function_value")
            obj_value = info[1]
        else:
            obj_value = None

        # Revert the delta

        # Restore bounds
        for rxn_id, (lb, ub) in original_bounds.items():
            var = self.variables[rxn_id]
            self.problem.changeColBounds(int(var), lb, ub)

        # Remove added variables and constraints
        # Delete in reverse index order so that index shifts don't invalidate
        # remaining highs_var/highs_cons objects. Base variables/constraints are
        # unaffected since they always have lower indices than added ones.
        for var in sorted(added_vars, key=lambda v: int(v), reverse=True):
            self.problem.deleteVariable(var, self.variables)

        for constr in sorted(added_constraints, key=lambda c: int(c), reverse=True):
            self.problem.removeConstr(constr)

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