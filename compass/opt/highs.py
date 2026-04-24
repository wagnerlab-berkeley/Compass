from compass.globals import BETA, EXCHANGE_LIMIT
from compass.models.MetabolicModel import MetabolicModel
from .base import Optimizer, LinearProgramDelta, Solution

import highspy

class HighsOptimizer(Optimizer):
    """
    Highs-based implementation of the Optimizer.
    """

    def __init__(self, model: MetabolicModel, config: dict[str, Any]):
        super().__init__(model)

        # TODO: Logging config?

        self.__init_base_model()

    def __create_base_model(self):
        """
        Builds the initial Highs model from the provided metabolic model.
        """
        problem = highspy.Highs()

        # Add reactions as variables
        variables = {}
        for id, reaction in self.model.reactions.items():
            lb = reaction.lower_bound
            ub = reaction.upper_bound
            var = self.problem.addVariable(lb=lb, ub=ub, name=reaction.id)
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
            constraint = self.problem.addConstr(0 == expr, name = metab_id)
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
            old_ub = lp.col_lower_[int(var)]
            old_lb = lp.col_upper_[int(var)]
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

        self.problem.run()

        model_status = self.problem.getModelStatus()
        info = self.problem.getInfo()
        if model_status == highspy.HighsModelStatus.kOptimal:
            success = True
        else:
            success = False

        # Revert the delta

        # Restore bounds
        for rxn_id, (lb, ub) in original_bounds.items():
            var = self.variables[rxn_id]
            self.problem.changeColBounds(int(var), lb, ub)

        # Remove added variables and constraints

        # Note that with Highs, variable/constraint numbering shifts downwards upon delete
        # But because the base variables predate these added vars, the base vars retain stable numbering.
        self.problem.deleteVars(len(added_vars), [int(x) for x in added_vars])

        for constr in added_constraints:
            self.problem.removeConstr(constr)

        return Solution(success=success, status=model_status, obj_value=info.objective_function_value)