import logging
from typing import Any
import multiprocessing

MULTIPROCESSING_CONFIGURED = False

def configure_multiprocessing():
    """
    Sets multiprocessing to use spawn instead of fork.
    This is because CUDA does not support forking, though for non-CUDA runs forking is
    generally faster.
    """
    global MULTIPROCESSING_CONFIGURED
    if not MULTIPROCESSING_CONFIGURED:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError as e:
            method = multiprocessing.get_start_method(allow_none=True)
            if method != "spawn":
             # Note for readers: CUDA init will fail if a process is forked.
             raise Exception(f"Multiprocessing start method was already set to {method}, but CUDA requires spawn") from e
        MULTIPROCESSING_CONFIGURED = True


configure_multiprocessing()

from cuopt.linear_programming.problem import Problem, Variable, Constraint, CONTINUOUS, MAXIMIZE, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings, PDLPSolverMode
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
    CUOPT_ABSOLUTE_GAP_TOLERANCE,
    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
    CUOPT_AUGMENTED,
    CUOPT_BARRIER_DUAL_INITIAL_POINT,
    CUOPT_CROSSOVER,
    CUOPT_CUDSS_DETERMINISTIC,
    CUOPT_DUAL_INFEASIBLE_TOLERANCE,
    CUOPT_DUAL_POSTSOLVE,
    CUOPT_DUALIZE,
    CUOPT_ELIMINATE_DENSE_COLUMNS,
    CUOPT_FIRST_PRIMAL_FEASIBLE,
    CUOPT_FOLDING,
    CUOPT_INFEASIBILITY_DETECTION,
    CUOPT_ITERATION_LIMIT,
    CUOPT_LOG_FILE,
    CUOPT_LOG_TO_CONSOLE,
    CUOPT_METHOD,
    CUOPT_MIP_ABSOLUTE_GAP,
    CUOPT_MIP_ABSOLUTE_TOLERANCE,
    CUOPT_MIP_HEURISTICS_ONLY,
    CUOPT_MIP_INTEGRALITY_TOLERANCE,
    CUOPT_MIP_RELATIVE_GAP,
    CUOPT_MIP_RELATIVE_TOLERANCE,
    CUOPT_MIP_SCALING,
    CUOPT_NUM_CPU_THREADS,
    CUOPT_ORDERING,
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_PER_CONSTRAINT_RESIDUAL,
    CUOPT_PRESOLVE,
    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
    CUOPT_RELATIVE_DUAL_TOLERANCE,
    CUOPT_RELATIVE_GAP_TOLERANCE,
    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    CUOPT_SAVE_BEST_PRIMAL_SO_FAR,
    CUOPT_SOLUTION_FILE,
    CUOPT_STRICT_INFEASIBILITY,
    CUOPT_TIME_LIMIT,
    CUOPT_USER_PROBLEM_FILE,
)

from compass.globals import EXCHANGE_LIMIT
from compass.models.MetabolicModel import MetabolicModel
from compass.opt.base import LinearProgramDelta, Optimizer, Solution

logger = logging.getLogger("compass")

def get_cuopt_config(threads: int | None = None, method: int | None = None) -> dict[str, Any]:
    """
    :param threads: Number of threads cuopt solver should use
    :type threads: int | None
    :param method: The method for solving linear programs
    :type method: int | None
    :return: Dictionary mapping cuopt settings to values
    :rtype: dict[str, Any]
    """
    if threads is None:
        threads = 1
    if method is None:
        method = 0
    return {
        # N.B. cuopt does not actually respect this flag, but should be resolved in coming PR
        CUOPT_LOG_TO_CONSOLE: False,
        CUOPT_PRESOLVE: False,
        CUOPT_NUM_CPU_THREADS: threads,
        # TODO: NVIDIA does not really explain these, but their docs indicate stable3 is generally the best
        CUOPT_PDLP_SOLVER_MODE: PDLPSolverMode.Stable3,
        # Based on C header here https://github.com/NVIDIA/cuopt/blob/7543358c9caca5e557d6e48fefa8d94baadcbab9/cpp/include/cuopt/linear_programming/constants.h#L104C1-L104C36
        # The methods are concurrent, pdlp, dual simplex, barrier enumerated as 0,1,2,3
        CUOPT_METHOD: method,
        # Determinism is preferrable for replication.
        CUOPT_CUDSS_DETERMINISTIC: True,
        # Each problem should take well under 120 seconds.
        CUOPT_TIME_LIMIT: 120,
    }


class CuoptOptimizer(Optimizer):
    """
    cuOpt-based implementation of the Optimizer
    """

    problem: Problem
    variables: dict[str, Variable]
    constraints: dict[str, Constraint]

    def __init__(self, model: MetabolicModel, config: dict[str, Any]):
        """
        Note that there is a default config, any values in the passed config will overwrite the default
        """
        super().__init__(model)
        self.solver_settings = SolverSettings()
        for name, value in config.items():
            self.solver_settings.set_parameter(name, value)

        # Initialize base model
        self.__init_base_model()

    def __create_base_model(self):
        """
        Note that cuOpt does not support deleting variables or constraints
        So we may need to call this multiple times
        """
        problem = Problem("compass_fba_problem")

        # Add reactions as variables
        variables = {}
        for id, reaction in self.model.reactions.items():
            lb = reaction.lower_bound
            ub = reaction.upper_bound
            var = problem.addVariable(lb=lb, ub=ub, name=reaction.id, vtype=CONTINUOUS)
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
            constraint = problem.addConstraint(expr == 0, name=metab_id)
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
        # Store original bounds to revert later
        original_ub = {}
        original_lb = {}
        # If we added vars or constraints, then the problem will be re-initialized
        # because cuOpt does not support deleting variables or constraints.
        added_vars = delta.added_secretion or delta.added_uptake

        for met_id, rxn_id in delta.added_secretion.items():
            rxn_var = self.problem.addVariable(lb=0.0, ub=self.model.maximum_flux, name=rxn_id, vtype=CONTINUOUS)
            self.variables[rxn_id] = rxn_var
            # Add secretion to metabolite's constraint as a reduction in metabolite
            met_constr = self.constraints[met_id]
            self.problem.updateConstraint(met_constr, coeffs=[(rxn_var, -1.0)])

        for met_id, rxn_id in delta.added_uptake.items():
            rxn_var = self.problem.addVariable(lb=0.0, ub=EXCHANGE_LIMIT, name=rxn_id, vtype=CONTINUOUS)
            self.variables[rxn_id] = rxn_var
            # Add uptake to metabolite's constraint as an increase in metabolite
            met_constr = self.constraints[met_id]
            self.problem.updateConstraint(met_constr, coeffs=[(rxn_var, 1.0)])

        for rxn_id in delta.blocked_reactions:
            var = self.variables[rxn_id]
            old_ub = var.getUpperBound()
            old_lb = var.getLowerBound()
            original_ub[rxn_id] = old_ub
            var.setUpperBound(old_lb)

        for rxn_id, limit in delta.high_flux.items():
            var = self.variables[rxn_id]
            old_lb = var.getLowerBound()
            original_lb[rxn_id] = old_lb
            var.setLowerBound(limit)

        objective_expr = sum([coeff * self.variables[id] for id, coeff in delta.objective.items()])
        if delta.sense == "max":
            sense = MAXIMIZE
        else:
            sense = MINIMIZE
        self.problem.setObjective(objective_expr, sense)

        try:
            self.solve_problem()
        except Exception as e:
            raise RuntimeError("Exception while solving cuOpt problem") from e

        status = self.problem.Status
        obj_value = self.problem.ObjValue
        if status.name == "Optimal":
            success = True
        else:
            success = False

        # Revert the delta (or rebuild the problem)
        if added_vars:
            self.__init_base_model()
        else:
            for rxn_id, ub in original_ub.items():
                self.variables[rxn_id].setUpperBound(ub)
            for rxn_id, lb in original_lb.items():
                self.variables[rxn_id].setLowerBound(lb)

        return Solution(success=success, status=status, obj_value=obj_value)
    
    def solve_problem(self):
        self.problem.solve(self.solver_settings)
        if self.problem.Status.name == "Optimal":
            return
        
        logger.info("Received non-optimal status %s. Retrying with automatic method selection", self.problem.Status.name)
        original_method = self.solver_settings.get_parameter(CUOPT_METHOD)

        # Retry with concurrent method, as that will try all methods
        try:
            self.solver_settings.set_parameter(CUOPT_METHOD, 0)
            self.problem.solve(self.solver_settings)
        finally:
            self.solver_settings.set_parameter(CUOPT_METHOD, original_method)
        return
        
