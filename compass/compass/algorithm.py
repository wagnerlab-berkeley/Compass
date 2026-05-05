"""
Run the procedure for COMPASS
"""
from __future__ import print_function, division, absolute_import
import pandas as pd
from tqdm import tqdm
from random import shuffle
import logging
import os
import sys
import time
import timeit
import numpy as np

from compass.models.MetabolicModel import MetabolicModel, Reaction
from compass.opt.base import LinearProgramDelta, Optimizer, Solution
from compass.opt.gurobi import GurobiOptimizer, get_gurobi_config
from .. import utils
from .. import models
from . import cache
from ..globals import BETA, EXCHANGE_LIMIT, LICENSE_DIR, MODEL_DIR
from .cache import PREPROCESS_CACHE_DIR
import compass.global_state as global_state

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger("compass")

__all__ = ['singleSampleCompass']

def singleSampleCompass(data, model, media, directory, sample_name, sample_index, args, metabolic_model_dir=MODEL_DIR, preprocess_cache_dir=PREPROCESS_CACHE_DIR):
    """
    Run Compass on a single column of data

    Parameters
    ==========
    data : list
       Full path to data file(s)

    model : str
        Name of metabolic model to use

    media : str or None
        Name of media to use

    directory : str
        Where to store results and log info.  Is created if it doesn't exist.

    sample_index : int
        Which sample to run on

    args : dict
        More keyword arguments
            - lambda, num_neighbors, symmetric_kernel, species,
              and_function, test_mode, detailed_perf
    """
    if not os.path.isdir(directory) and directory != '/dev/null':
        os.makedirs(directory)

    if os.path.exists(os.path.join(directory, 'success_token')):
        logger.info('success_token detected, results already calculated.')
        logger.info('COMPASS Completed Successfully')
        return

    if args['save_argmaxes']:
        args['save_argmaxes_dir'] = directory
    else:
        args['save_argmaxes_dir'] = None

    model = models.init_model(model=model, species=args['species'],
                       exchange_limit=EXCHANGE_LIMIT, media=args['media'], 
                       isoform_summing=args['isoform_summing'], metabolic_model_dir=metabolic_model_dir)

    logger.info("Running COMPASS on model: %s", model.name)

    perf_log = None
    if args['detailed_perf']:
        cols = ['order','max rxn time', 'max rxn method', 'cached', 'min penalty time', 
        'min penalty method', 'min penalty sensitvivity', 'kappa']
        perf_log = {c:{} for c in cols}

    if args['generate_cache']:
        cache.clear(model, preprocess_cache_dir=preprocess_cache_dir) #TBD add media specifier here too

    # Build model into Gurobi model
    opt = initialize_optimization(model, args)
    
    logger.info(f'Processing Sample {sample_index}: {sample_name}')
    global_state.set_current_cell_name(sample_name)

    # Run core compass algorithm

    # Read in reaction penalties
    logger.info("Reading Reaction Penalties...")
    reaction_penalties = pd.read_csv(
        os.path.join(args['penalties_dir'], f'sample{sample_index}', 'penalties.txt.gz'), index_col=0, sep='\t').iloc[:, 0]

    reaction_penalties_dict = dict(reaction_penalties.items())
    react_start = time.process_time()
    if not args['no_reactions']:
        logger.info("Evaluating Reaction Scores...")
        reaction_scores = compass_reactions(
            model, opt=opt, reaction_penalties=reaction_penalties_dict,
            perf_log=perf_log, args=args, preprocess_cache_dir=preprocess_cache_dir)
    react_elapsed = time.process_time() - react_start

    #if user wants to calc reaction scores, but doesn't want to calc metabolite scores, calc only the exchange reactions
    logger.info("Evaluating Exchange/Secretion/Uptake Scores...")
    exchange_start = time.process_time()
    uptake_scores, secretion_scores, exchange_rxns = compass_exchange(
        model, opt=opt, reaction_penalties=reaction_penalties_dict,
        only_exchange=(not args['no_reactions']) and not args['calc_metabolites'],
        perf_log=perf_log, args=args, preprocess_cache_dir=preprocess_cache_dir)
    exchange_elapsed = time.process_time() - exchange_start

    # Copy valid uptake/secretion reaction fluxes from uptake/secretion
    #   results into reaction results
    if (not args['no_reactions']) or args['calc_metabolites']:
        for r_id in exchange_rxns:
            assert r_id in model.reactions
            assert r_id not in reaction_scores
            reaction_scores[r_id] = exchange_rxns[r_id]

    # Output results to file
    logger.info("Writing output files...")
    if not args['no_reactions']:
        reaction_scores = pd.Series(reaction_scores, name=sample_name).sort_index()
        reaction_scores.to_csv(os.path.join(directory, 'reactions.txt'),
                               sep="\t", header=True)

    if args['calc_metabolites']:
        uptake_scores = pd.Series(uptake_scores, name=sample_name).sort_index()
        secretion_scores = pd.Series(secretion_scores, name=sample_name).sort_index()

        uptake_scores.to_csv(os.path.join(directory, 'uptake.txt'),
                             sep="\t", header=True)
        secretion_scores.to_csv(os.path.join(directory, 'secretions.txt'),
                                sep="\t", header=True)

    if args['generate_cache'] or cache.is_new_cache(model):
        logger.info(
            'Saving cache file for Model: {}, Media: {}'.format(
                model.name, model.media)
        )
        cache.save(model, preprocess_cache_dir=preprocess_cache_dir)

    # write success token
    with open(os.path.join(directory, 'success_token'), 'w') as fout:
        fout.write('Success!')

    if not args['no_reactions']:
        logger.info("Compass Reaction Time: "+str(react_elapsed))
        logger.info("Processed "+str(len(reaction_scores))+" reactions")
    logger.info("Compass Exchange Time: "+str(exchange_elapsed))
    logger.info("Processed "+str(len(uptake_scores))+" uptake reactions")
    logger.info("Processed "+str(len(secretion_scores))+" secretion reactions")
    
    if perf_log is not None:
        perf_log = pd.DataFrame(perf_log)
        perf_log.to_csv(os.path.join(directory, "compass_performance_log.csv"))
        logger.info("Saved detailed performance log")
    
    logger.info('COMPASS Completed Successfully')
    
def read_selected_reactions(select_reactions, select_subsystems, model):
    selected_reaction_ids = []
    if select_reactions:
        if not os.path.exists(select_reactions):
            raise Exception("cannot find selected reactions subset file %s" % select_reactions)
        with open(select_reactions) as f:
            selected_reaction_ids += [line.strip() for line in f]
    if select_subsystems:
        if not os.path.exists(select_subsystems):
            raise Exception("cannot find selected reactions subset file %s" % select_subsystems)
        with open(select_subsystems) as f:
            selected_subsystems_ids = [line.strip() for line in f]
        subsys = {}
        for rxn in model.reactions.values():
            if rxn.subsystem in selected_subsystems_ids:
                selected_reaction_ids += [rxn.id]
    return [str(s) for s in selected_reaction_ids]


def compass_exchange(
        model: MetabolicModel, 
        opt: Optimizer, 
        reaction_penalties: dict[str, float], 
        only_exchange=False, 
        perf_log=None, 
        args = None, 
        preprocess_cache_dir=PREPROCESS_CACHE_DIR
    ):
    """
    Iterates through metabolites, finding each's max
    uptake and secretion potentials. If only_exchange=True, does so only for exchange reactions.

    Holds each near its max uptake/secretion while minimizing
    penalty

    Returns the optimal penalty for uptake and secretion

    Returns
    -------
    uptake_scores: dict
        key: species_id
        value: minimum penalty achieved

    secretion_scores: dict
        key: species_id
        value: minimum penalty achieved

    exchange_rxns: dict
        Separate storage for exchange reactions.  These
        are skipped in the reaction loop.
        key: rxn_id
        value: minimum penalty achieved
    """

    # Setting only_exchange=False might cause a pair of uptake/secretion reactions associated with a certain
    # metabolite to be added even if there was no uptake/secretion reaction associated with this metabolite
    # only_exchange=True adds the other half of the pair if there is an uptake/secretion reaction and does not
    # if there is no such reaction

    secretion_scores = {}
    uptake_scores = {}
    exchange_rxns = {}
    metabolites = list(model.species.values())
    if args['test_mode']:
        metabolites = metabolites[0:50]

    #populate the list of selected_reaction_ids - do this once outside of the loop
    if args['select_reactions'] or args['select_subsystems']:
        selected_reaction_ids = read_selected_reactions(args['select_reactions'], args['select_subsystems'], model)

    for metabolite in tqdm(metabolites, file=sys.stderr):

        met_id = metabolite.id
        metabolite_used = model.is_metabolite_used(metabolite_id=met_id)

        if not metabolite_used:
            # This can happen if the metabolite does not participate
            # in any reaction. As a result, it won't be in any
            # constraints - happens in RECON2

            uptake_scores[met_id] = 0.0
            secretion_scores[met_id] = 0.0
            continue

        # Rectify exchange reactions
        # Either find existing pos and neg exchange reactions
        # Or create new ones

        uptake_rxn = None
        extra_uptake_rxns = []
        secretion_rxn = None
        extra_secretion_rxns = []

        added_uptake = {}     # Did we add an uptake reaction?
        added_secretion = {}  # "   "   "  "  secretion reaction?

        # Metabolites represented by a constraint: get associated reactions
        rxn_ids = model.associated_reactions(met_id)
        reactions = [model.reactions[x] for x in rxn_ids]

        #If user wants only exchange reaction - limit the reactions space through which we iterate
        if only_exchange:
            reactions = [x for x in reactions if x.is_exchange]

        # Check if any reactions are selected at all
        if args['select_reactions'] or args['select_subsystems']:
            #r.id is a unidirectional identifier (ending with _pos or _neg suffix --> we remove it and compare to the undirected reaction id)
            reactions = [r for r in reactions if ((r.id)[:-4] in selected_reaction_ids or str(r.id) in selected_reaction_ids)]

        # If there are no selected, then go to next metabolite
        if not reactions:
            logger.debug(f"Skipping {met_id}")
            continue
        
        # If we care about any of these reactions, then collect optimal uptake and secretion.
        # NOTE: This may include more reactions than strictly selected
        reactions = [model.reactions[x] for x in rxn_ids]

        # Extra reactions are duplicates
        for reaction in reactions:
            if reaction.is_exchange and met_id in reaction.products:
                if uptake_rxn is None:
                    uptake_rxn = reaction.id
                else:
                    extra_uptake_rxns.append(reaction.id)

            elif reaction.is_exchange and met_id in reaction.reactants:
                if secretion_rxn is None:
                    secretion_rxn = reaction.id
                else:
                    extra_secretion_rxns.append(reaction.id)

        #if the selected_rxns or only_exchange options are used --> then we don't want to add reactions unless one of the pair already exists
        if (only_exchange or args['select_reactions']) and (uptake_rxn is None) and (secretion_rxn is None):
            continue
        
        if secretion_rxn is None:
            secretion_rxn = met_id + "_SECRETION"
            added_secretion[met_id] = secretion_rxn

        if uptake_rxn is None:
            uptake_rxn = met_id + "_UPTAKE"
            added_uptake[met_id] = uptake_rxn

        # Modify the constraint in the problem
        #   e.g. Add the metabolites connections
        all_uptake = [uptake_rxn] + extra_uptake_rxns
        all_secretion = [secretion_rxn] + extra_secretion_rxns

        # -----------------
        # Optimal Secretion
        # -----------------
        
        # Close all uptake and extra secretion
        blocked_reactions = all_uptake + extra_secretion_rxns
        # Get max of secretion reaction
        secretion_max = maximize_reaction(
            model, 
            opt, 
            secretion_rxn, 
            perf_log=perf_log, 
            preprocess_cache_dir=preprocess_cache_dir, 
            blocked_reactions=blocked_reactions, 
            added_secretion=added_secretion, 
            added_uptake=added_uptake
        )

        # Constrain secretion to be at least BETA * v_r^opt
        high_flux = { secretion_rxn: BETA * secretion_max }

        # Minimize Penalty
        delta = LinearProgramDelta(
            objective=reaction_penalties, 
            sense="min", 
            added_secretion=added_secretion, 
            added_uptake=added_uptake,
            blocked_reactions=blocked_reactions, 
            high_flux=high_flux
        )
        
        if perf_log is not None:
            start_time = time.process_time()

        global_state.set_current_reaction_id(secretion_rxn)
        sol = solve_model_wrapper(opt, delta)
        secretion_scores[met_id] = sol.obj_value
        
        if perf_log is not None:
            perf_log['min penalty time'][secretion_rxn] = time.process_time() - start_time

        # -----------------
        # Optimal Uptake
        # -----------------
        
        # Close extra uptake and all secretion
        blocked_reactions = extra_uptake_rxns + all_secretion
        # Get max of uptake reaction
        uptake_max = maximize_reaction(
            model, 
            opt, 
            uptake_rxn, 
            perf_log=perf_log, 
            preprocess_cache_dir=preprocess_cache_dir,
            blocked_reactions=blocked_reactions, 
            added_secretion=added_secretion, 
            added_uptake=added_uptake
        )

        # Constrain uptake to be at least BETA * _r^opt
        high_flux = { uptake_rxn: BETA * uptake_max }

        # Minimize Penalty
        delta = LinearProgramDelta(
            objective=reaction_penalties, 
            sense="min", 
            added_secretion=added_secretion, 
            added_uptake=added_uptake,
            blocked_reactions=blocked_reactions, 
            high_flux=high_flux
        )

        if perf_log is not None:
            start_time = time.process_time()

        global_state.set_current_reaction_id(uptake_rxn)
        sol = solve_model_wrapper(opt, delta)
        uptake_scores[met_id] = sol.obj_value

        if perf_log is not None:
            perf_log['min penalty time'][uptake_rxn] = time.process_time() - start_time

        # For reactions that were not artificially added, update the exchange_rxns dict
        if not added_uptake:
            for rxn_id in all_uptake:
                exchange_rxns[rxn_id] = uptake_scores[met_id]

        if not added_secretion:
            for rxn_id in all_secretion:
                exchange_rxns[rxn_id] = secretion_scores[met_id]
            
    return uptake_scores, secretion_scores, exchange_rxns

def compass_reactions(
        model: MetabolicModel, 
        opt: Optimizer, 
        reaction_penalties: dict[str, float], 
        perf_log=None, 
        args = None, 
        preprocess_cache_dir=PREPROCESS_CACHE_DIR
    ):

    """
    Iterates through reactions, holding each near
    its max value while minimizing penalty.

    Minimum overall penalty returned for each reaction

    Returns
    -------
    reaction_scores: dict
        key: reaction id
        value: minimum penalty achieved
    """
    # Iterate through Reactions

    reaction_scores = {}
    
    reactions = list(model.reactions.values())

    if args['test_mode']:
        reactions = reactions[0:100]

    if args['select_reactions'] or args['select_subsystems']:
        selected_reaction_ids = read_selected_reactions(args['select_reactions'], args['select_subsystems'], model)
        #r.id is a unidirectional identifier (ending with _pos or _neg suffix --> we remove it and compare to the undirected reaction id)
        
        reactions = [r for r in reactions if (str(r.id)[:-4] in selected_reaction_ids or str(r.id) in selected_reaction_ids)]

    for reaction in tqdm(reactions, file=sys.stderr):

        if reaction.is_exchange:
            continue
        
        # Logic for blocking partner reaction now in maximize_reaction
        r_max = maximize_reaction(model, opt, reaction.id, perf_log=perf_log, preprocess_cache_dir=preprocess_cache_dir)

        # If Reaction can't carry flux anyways (v_r^opt = 0), just continue
        if r_max == 0:
            reaction_scores[reaction.id] = 0
            if perf_log is not None:
               perf_log['min penalty time'][reaction.id] = 0
               #perf_log['blocked'][reaction.id] = True

        else:
            
            # Block reverse reaction
            blocked_reactions = []
            if reaction.reverse_reaction:
                blocked_reactions.append(reaction.reverse_reaction.id)

            # Constrain reaction to be at least BETA * v_r^opt
            high_flux = { reaction.id: BETA * r_max }

            # Minimize Penalty
            delta = LinearProgramDelta(
                objective=reaction_penalties, 
                sense="min", 
                blocked_reactions=blocked_reactions, 
                high_flux=high_flux
            )

            if perf_log is not None:
                #perf_log['blocked'][reaction.id] = False
                start_time = time.process_time()

            global_state.set_current_reaction_id(reaction.id)
            sol = solve_model_wrapper(opt, delta)
            reaction_scores[reaction.id] = sol.obj_value

            # TODO: modify for gurobi
            if perf_log is not None:
                perf_log['min penalty time'][reaction.id] = time.process_time() - start_time
                #perf_log['min penalty method'][reaction.id] = gp_model.getParamInfo('Method')
                #perf_log['min penalty sensitvivity'][reaction.id] = problem.solution.sensitivity.objective(reaction.id)
                #if hasattr(problem.solution.get_quality_metrics(),'kappa'):
                   #perf_log['kappa'][reaction.id] = problem.solution.get_quality_metrics().kappa

            #if args['save_argmaxes']:
            #    gp_model.write(os.path.join(args['save_argmaxes_dir'], f'{reaction.id}.sol'))

    return reaction_scores

def initialize_optimization(model: MetabolicModel, args) -> Optimizer:
    """
    Builds a flux balance analysis model from the specified metabolic model.
    Uses args to determine which optimizer will be used and how to configure it.
    """
    num_threads = args.get('num_threads')
    lpmethod = args.get('lpmethod')
    if args['optimizer'] == "gurobi":
        credentials = utils.get_gurobi_credentials()
        config = get_gurobi_config(num_threads, lpmethod)
        return GurobiOptimizer(model, config, credentials=credentials)
    elif args['optimizer'] == "cuopt":
        try:
            from compass.opt.cuopt import CuoptOptimizer, get_cuopt_config
        except ImportError as e:
            raise ImportError(
                "The cuopt optimizer requires NVIDIA cuOpt, which is not installed. "
                "cuOpt requires a GPU and an NVIDIA NGC API key. "
                "See https://developer.nvidia.com/cuopt-logistics-optimization for details."
            ) from e
        config = get_cuopt_config(num_threads, method=lpmethod)
        return CuoptOptimizer(model, config)
    elif args['optimizer'] == "highs":
        from compass.opt.highs import HighsOptimizer, get_highs_config
        config = get_highs_config(num_threads, method=lpmethod)
        return HighsOptimizer(model, config)

def maximize_reaction(
        model: MetabolicModel, 
        opt: Optimizer, 
        rxn_id: str, 
        use_cache=True, 
        perf_log=None, 
        preprocess_cache_dir=PREPROCESS_CACHE_DIR,
        blocked_reactions: list[str] | None = None,
        added_secretion: dict | None = None,
        added_uptake: dict | None = None,
    ):
    """
    Maximizes the current reaction in the problem
    Attempts to retrieve the value from cache, and populates the cache on a cache miss
    The blocked_reactions, added_secretion, and added_uptake parameters are used for compass_exchange
    """
    if blocked_reactions is None:
        blocked_reactions = []
    if added_secretion is None:
        added_secretion = {}
    if added_uptake is None:
        added_uptake = {}

    if perf_log is not None:
        start_time = time.process_time()
        perf_log['order'][rxn_id] = len(perf_log['order'])

    # Load from cache if it exists and return
    if use_cache:
        model_cache = cache.load(model, preprocess_cache_dir=preprocess_cache_dir)
        if rxn_id in model_cache:
            if perf_log is not None:
                perf_log['cached'][rxn_id] = True
                perf_log['max rxn time'][rxn_id] = time.process_time() - start_time
            return model_cache[rxn_id]

    # Set partner reaction upper-limit to 0 in problem, if it exists
    rxn = model.reactions.get(rxn_id)
    if rxn.reverse_reaction:
        blocked_reactions.append(rxn_id.reverse_reaction.id)

    # Maximize the reaction
    objective = { rxn_id : 1.0 }
    delta = LinearProgramDelta(
        objective=objective, 
        sense="max", 
        blocked_reactions=blocked_reactions, 
        added_secretion=added_secretion, 
        added_uptake=added_uptake
    )
    
    global_state.set_current_reaction_id(rxn_id)
    sol = solve_model_wrapper(opt, delta)

    # Save the result
    model_cache = cache.load(model, preprocess_cache_dir=preprocess_cache_dir)
    model_cache[rxn_id] = sol.obj_value

    if perf_log is not None:
        perf_log['cached'][rxn_id] = False
        perf_log['max rxn time'][rxn_id] = time.process_time() - start_time
        # TODO: modify for gurobi/cuopt
        #perf_log['max rxn method'][rxn] = problem.solution.get_method()
        #perf_log['max rxn method'][rxn] = gp_model.getParamInfo('Method')

    return sol.obj_value

def maximize_reaction_range(start_stop, args, model_name=None, metabolic_model_dir=MODEL_DIR):
    """
    Maximizes a range of reactions from start_stop=(start, stop).
    args must be a dict with keys 'model', 'species', 'media'

    This is to reduce overhead compared to initializing the model and problem each time
    The model and problem cannot be passed to partial because they are not pickleable
    and pool requires the function to be pickleable.

    Returns
    -------
    sub_cache : dict
        key : reaction id
        value : maximum flux for reaction
    """

    if model_name is None:
        model_name = args['model']

    #make a sub cache for each thread to write into
    sub_cache = {}

    model = models.init_model(model=model_name, species=args['species'],
                       exchange_limit=EXCHANGE_LIMIT, media=args['media'], 
                       isoform_summing=args['isoform_summing'], metabolic_model_dir=metabolic_model_dir)
    opt = initialize_optimization(model, args)

    #sort by id to ensure consistency across threads
    reactions = sorted(list(model.reactions.values()), key=lambda r:r.id)[start_stop[0]:start_stop[1]]
    for reaction in tqdm(reactions, file=sys.stderr):
        #if reaction.is_exchange:
        #    continue
        blocked_reactions = []
        partner_reaction = reaction.reverse_reaction

        # Set partner reaction upper-limit to 0 in problem
        if partner_reaction is not None:
            blocked_reactions.append(partner_reaction.id)

        objective = { reaction.id : 1.0 }
        delta = LinearProgramDelta(
            objective=objective, 
            sense="max", 
            blocked_reactions=blocked_reactions, 
            added_secretion={}, 
            added_uptake={}
        )
        sol = solve_model_wrapper(opt, delta)

        sub_cache[reaction.id] = sol.obj_value

    return sub_cache

def maximize_metab_range(start_stop, args, model_name=None, metabolic_model_dir=MODEL_DIR):
    """
    For precaching the maxmimum feasible exchange values of metabolites

    Returns
    -------
    sub_cache: dict
        key: species_id
        value: maximum flux
    """

    if model_name is None:
        model_name=args['model']

    sub_cache = {}
    model = models.init_model(model=model_name, species=args['species'],
                       exchange_limit=EXCHANGE_LIMIT, media=args['media'], 
                       isoform_summing=args['isoform_summing'], metabolic_model_dir=metabolic_model_dir)
    opt = initialize_optimization(model, args)

    # init_model returns entire list of metabolites as specified by the RECON2 model
    # However, some metabolites are not associated with any reactions,
    # and maximize_metab_range only computes those that are associated with reactions
    # Therefore model.species might contain metabolites that don't need to be maximized at all
    metabolites = sorted(list(model.species.values()), key=lambda m:m.id)[start_stop[0]:start_stop[1]]

    for metabolite in tqdm(metabolites, file=sys.stderr):

        met_id = metabolite.id
        metabolite_used = model.is_metabolite_used(metabolite_id=met_id)

        if not metabolite_used:
            # This can happen if the metabolite does not participate
            # in any reaction. As a result, it won't be in any
            # constraints - happens in RECON2
            continue

        # Rectify exchange reactions
        # Either find existing pos and neg exchange reactions
        # Or create new ones

        uptake_rxn = None
        extra_uptake_rxns = []
        secretion_rxn = None
        extra_secretion_rxns = []

        # Maps the metabolite id to the name of the new reaction
        added_uptake = {}     # Did we add an uptake reaction?
        added_secretion = {}  # "   "   "  "  secretion reaction?

        # Metabolites represented by a constraint: get associated reactions
        rxn_ids = model.associated_reactions(met_id)
        reactions = [model.reactions[x] for x in rxn_ids]

        # NOTE: only_exchange flag does not apply to maximize_metab_range
        # however this has no effect since filtering for exchange reactions
        # is still done within 'for reaction in reactions' loop

        # if only_exchange:
        #     reactions = [x for x in reactions if x.is_exchange]

        for reaction in reactions:
            if reaction.is_exchange and met_id in reaction.products:
                if uptake_rxn is None:
                    uptake_rxn = reaction.id
                else:
                    extra_uptake_rxns.append(reaction.id)

            elif reaction.is_exchange and met_id in reaction.reactants:
                if secretion_rxn is None:
                    secretion_rxn = reaction.id
                else:
                    extra_secretion_rxns.append(reaction.id)

        # NOTE: only_exchange flag does not apply to maximize_metab_range
        # i.e. pairs of exchange reactions can be added even if neither exist

        #if the selected_rxns or only_exchange options are used --> then we don't want to add reactions unless one of the pair already exists

        if secretion_rxn is None:
            secretion_rxn = met_id + "_SECRETION"
            added_secretion[met_id] = secretion_rxn

        if uptake_rxn is None:
            uptake_rxn = met_id + "_UPTAKE"
            added_uptake[met_id] = uptake_rxn

        # Modify the constraint in the problem
        #   e.g. Add the metabolites connections
        all_uptake = [uptake_rxn] + extra_uptake_rxns
        all_secretion = [secretion_rxn] + extra_secretion_rxns

        # -----------------
        # Optimal Secretion
        # -----------------

        # Close all uptake and extra secretion
        blocked_reactions = all_uptake + extra_secretion_rxns

        # Get max of secretion reaction
        objective = { secretion_rxn : 1.0 }
        delta = LinearProgramDelta(
            objective=objective, 
            sense="max", 
            blocked_reactions=blocked_reactions, 
            added_secretion=added_secretion, 
            added_uptake=added_uptake,
        )
        sol = solve_model_wrapper(opt, delta)

        # NOTE: since only_exchange flag is not applicable here, additional secretion reactions
        # can be created and added to the cache
        sub_cache[secretion_rxn] = sol.obj_value

        # -----------------
        # Optimal Uptake
        # -----------------

        # Close extra uptake and all secretion
        blocked_reactions = extra_uptake_rxns + all_secretion

        # Get max of uptake reaction
        objective = { uptake_rxn : 1.0 }
        delta = LinearProgramDelta(
            objective=objective, 
            sense="max", 
            blocked_reactions=blocked_reactions, 
            added_secretion=added_secretion, 
            added_uptake=added_uptake,
        )
        sol = solve_model_wrapper(opt, delta)

        # NOTE: since only_exchange flag is not applicable here, additional uptake reactions
        # can be created and added to the cache
        sub_cache[uptake_rxn] = sol.obj_value
            
    return sub_cache
    
def solve_model_wrapper(opt: Optimizer, delta: LinearProgramDelta) -> Solution:
    r"""
    Only optimize the model if the reaction is selected for the cell. Else,
    skip the computation and return np.nan as the optimal value
    """
    if global_state.current_reaction_is_selected_for_current_cell():
        sol = opt.solve(delta)
        if not sol.success:
            logger.error("Failed to solve linear program: %s", sol.status)
        return sol
    else:
        return Solution(success=True, status="Skipped", obj_value=np.nan)