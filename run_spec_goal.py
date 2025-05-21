from typing import List, cast, Dict, Any, Optional

import torch

from deepxube.environments.environment_abstract import EnvGrndAtoms, State, Goal, Action
from deepxube.search.astar import AStar, Node, get_path
from deepxube.utils import viz_utils, data_utils
from deepxube.environments.env_utils import get_environment
from deepxube.nnet import nnet_utils
from deepxube.nnet.nnet_utils import HeurFN_T
from deepxube.logic.logic_objects import Clause
from deepxube.logic.logic_utils import parse_clause
from deepxube.logic.asp import parse_clingo_line
from deepxube.specification.spec_goal_asp import SpecSearchASP, RefineArgs, PathSoln, PathFn
from argparse import ArgumentParser
import os
import sys
import pickle
import numpy as np
import time


def get_heur_fn(env: EnvGrndAtoms, heur_fn_file: str, nnet_batch_size: int, num_procs: int) -> HeurFN_T:
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))
    heur_fn: HeurFN_T = nnet_utils.load_heuristic_fn(heur_fn_file, device, on_gpu, env.get_v_nnet(), env,
                                                     clip_zero=True, batch_size=nnet_batch_size)
    if not ('CUDA_VISIBLE_DEVICES' in os.environ) or not torch.cuda.is_available():
        torch.set_num_threads(num_procs)

    return heur_fn


def get_astar_path_fn(heur_fn: HeurFN_T, path_batch_size: int, weight: float, max_itrs: int,
                      search_verbose: bool) -> PathFn:
    def path_fn(env: EnvGrndAtoms, states_start: List[State], goals: List[Goal]) -> List[Optional[PathFn]]:
        astar = AStar(env)
        astar.add_instances(states_start, goals, [weight] * len(goals), heur_fn)

        # search
        search_itr: int = 0
        while (not min(x.finished for x in astar.instances)) and (search_itr < max_itrs):
            search_itr += 1
            astar.step(heur_fn, path_batch_size, verbose=search_verbose)

        goal_nodes: List[Optional[Node]] = [x.goal_node for x in astar.instances]

        path_solns: List[Optional[PathSoln]] = []
        for goal_node in goal_nodes:
            if goal_node is None:
                path_solns.append(None)
            else:
                path_states, path_actions, path_cost = get_path(goal_node)
                path_solns.append(PathSoln(path_states, path_actions, path_cost))

        return path_solns

    return path_fn


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    # environment, background knowledge
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--bk_add', type=str, default=None, help="File of additional background knowledge")

    # heuristic function
    parser.add_argument('--heur', type=str, required=True, help="nnet model file")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect path found, but will "
                                                                          "help if nnet is running out of memory.")
    parser.add_argument('--heur_procs', type=int, default=8, help="Number of parallel CPUs if using heur_fn on CPU")

    # specification, data, and results
    parser.add_argument('--spec', type=str, required=True, help="Should have 'goal' in the head. "
                                                                "Separate multiple clauses by ';'")
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")
    parser.add_argument('--start_idx', type=int, default=-1, help="Manually set on which example to start")
    parser.add_argument('--results', type=str, required=True, help="Directory to save results. Saves results after "
                                                                   "every instance.")
    parser.add_argument('--redo', action='store_true', default=False, help="Set to start from scratch")

    # refinement
    parser.add_argument('--refine_rand', action='store_true', default=False, help="")
    parser.add_argument('--refine_conf', action='store_true', default=False, help="")
    parser.add_argument('--refine_confkeep', action='store_true', default=False, help="")

    # search
    parser.add_argument('--expand_size', type=int, default=1, help="Maximum number of models for expansion")
    parser.add_argument('--ub_pat', type=int, default=-1, help="Number of iterations to wait to upper bound to improve "
                                                               "after finding a solution. -1 is to wait until queue is "
                                                               "empty")

    # pathfinding
    parser.add_argument('--path_batch_size', type=int, default=100, help="Batch size for batch-weighted A* search")
    parser.add_argument('--weight', type=float, default=0.2, help="Weight on path cost f(n) = w * g(n) + h(n)")
    parser.add_argument('--max_path_search_itrs', type=float, default=100, help="Maximum number of iterations to "
                                                                                "search for a path to a given model.")

    # verbosity, visualization, and debugging
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose search")
    parser.add_argument('--path_verbose', action='store_true', default=False, help="Set for verbose pathfinding")
    parser.add_argument('--viz_start', action='store_true', default=False, help="Set to visualize starting state")
    parser.add_argument('--viz_model', action='store_true', default=False, help="Set to visualize each model before "
                                                                                "search")
    parser.add_argument('--viz_conf', action='store_true', default=False, help="Set to visualize conflict")
    parser.add_argument('--viz_reached', action='store_true', default=False, help="Set to visualize reached goal state")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging with breakpoints")

    parser.add_argument('--time_limit', type=float, default=-1.0, help="A time limit for search. Default is -1, "
                                                                       "which means infinite.")

    args = parser.parse_args()

    # Directory
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    # environment
    env: EnvGrndAtoms = cast(EnvGrndAtoms, get_environment(args.env))
    # states: List[State] = env.get_start_states(10)
    # env.model_to_goal(env.state_to_model(states))

    bk_file_name: str = f"{args.results}/bk.lp"
    bk: List[str] = env.get_bk()
    with open(bk_file_name, "w") as bk_file:
        for bk_line in bk:
            bk_line = parse_clingo_line(bk_line)
            bk_file.write(f"{bk_line}\n")

    # get data
    data: Dict = pickle.load(open(args.states, "rb"))
    states: List[State] = data['states']

    results_file: str = "%s/results.pkl" % args.results
    output_file: str = "%s/output.txt" % args.results

    has_results: bool = False
    if os.path.isfile(results_file):
        has_results = True

    if has_results and (not args.redo):
        results: Dict[str, Any] = pickle.load(open(results_file, "rb"))
        log_write_mode: str = "a"
    else:
        results: Dict[str, Any] = {"states": states, "actions": [], "path_costs": [], "solved": [], "itrs": [],
                                   "num_models_gen": [], "per_reached": [], "per_not_goal": [], "secs/model": [],
                                   "secs/path": [], "times_tot": [], "times": [], "stats": []}
        log_write_mode: str = "w"

    if (not args.debug) and (not (args.viz_start or args.viz_model or args.viz_conf or args.viz_reached)):
        sys.stdout = data_utils.Logger(output_file, log_write_mode)

    # spec clauses
    spec_clauses_str = args.spec.split(";")
    clauses: List[Clause] = []
    for clause_str in spec_clauses_str:
        clause = parse_clause(clause_str)[0]
        clauses.append(clause)
    print("Parsed input clauses:")
    print(clauses)

    heur_fn: HeurFN_T = get_heur_fn(env, args.heur, args.nnet_batch_size, args.heur_procs)
    path_fn: PathFn = get_astar_path_fn(heur_fn, args.path_batch_size, args.weight, args.max_path_search_itrs,
                                        args.path_verbose)

    # find paths
    if args.start_idx >= 0:
        start_idx = args.start_idx
    else:
        start_idx = len(results["actions"])

    for state_idx in range(start_idx, len(states)):
        # start state
        state = states[state_idx]
        if args.viz_start:
            print("Start")
            viz_utils.visualize_examples(env, [state])

        start_time = time.time()
        refine_args: RefineArgs = RefineArgs(args.refine_rand, args.refine_conf, args.refine_confkeep, args.expand_size)
        search: SpecSearchASP = SpecSearchASP(env, state, clauses, path_fn, refine_args, bk_add=args.bk_add,
                                              verbose=args.verbose, viz_model=args.viz_model, viz_conf=args.viz_conf,
                                              viz_reached=args.viz_reached)

        itr: int = 0
        ub_curr: float = search.ub
        ub_patience: int = 0
        while not search.is_terminal():
            search.step()
            itr += 1
            if search.ub < ub_curr:
                ub_curr = search.ub
                ub_patience = 0
            elif ub_curr < np.inf:
                ub_patience += 1

            if (args.time_limit >= 0) and ((time.time() - start_time) > args.time_limit):
                break
            if (args.ub_pat >= 0) and (ub_patience >= args.ub_pat):
                break

        time_tot = time.time() - start_time

        not_seen_tot: int = sum(x['#not_seen'] for x in search.stats)
        path_reached_tot: int = sum(x['#reached'] for x in search.stats)
        not_goal_tot: int = sum(x['#reached_not_goal'] for x in search.stats)
        per_reached: float = 100 * path_reached_tot / max(not_seen_tot, 1)
        per_not_goal: float = 100 * not_goal_tot / max(path_reached_tot, 1)

        secs_model: float = search.times.times["refine"] / max(search.num_models_gen, 1)
        secs_path: float = search.times.times["path_find"] / max(not_seen_tot, 1)

        path_actions: Optional[List[Action]] = None
        path_cost: float = np.inf
        solved: bool = False
        if search.soln_best is not None:
            path_actions: List[Any] = search.soln_best.path_actions
            path_cost: float = search.soln_best.path_cost
            solved = True

        results["actions"].append(path_actions)
        results["path_costs"].append(path_cost)
        results["solved"].append(solved)
        results["itrs"].append(itr)
        results["num_models_gen"].append(search.num_models_gen)
        results["per_reached"].append(per_reached)
        results["per_not_goal"].append(per_not_goal)
        results["secs/model"].append(secs_model)
        results["secs/path"].append(secs_path)
        results["stats"].append(search.stats)
        results["times_tot"].append(time_tot)
        results["times"].append(search.times)

        print(f"State: {state_idx}, path_cost: {path_cost:.2f}, solved: {solved}, #itrs: {itr}, "
              f"#models: {search.num_models_gen}, %reach: {per_reached:.2f}, %not_goal: {per_not_goal:.2f}, "
              f"secs/model: {secs_model:.2f}, secs/path: {secs_path:.2f}, "
              f"Time: {time_tot:.2f}")
        print(f"Times - {search.times.get_time_str()}")
        print(f"Means - path_cost: {_get_mean(results, 'path_costs'):.2f}, "
              f"solved: {100.0 * np.mean(results['solved']):.2f}%, #itrs: {_get_mean(results, 'itrs'):.2f}, "
              f"#models: {_get_mean(results, 'num_models_gen'):.2f}, "
              f"%reach: {_get_mean(results, 'per_reached'):.2f}, %not_goal: {_get_mean(results, 'per_not_goal'):.2f}, "
              f"secs/model: {_get_mean(results, 'secs/model'):.2f}, secs/path: {_get_mean(results, 'secs/path'):.2f}, "
              f"Time: {_get_mean(results, 'times_tot'):.2f}")
        print("")

        if solved and args.viz_reached:
            print("Goal")
            path_states: List[State] = search.soln_best.path_states
            viz_utils.visualize_examples(env, [path_states[-1]])
        pickle.dump(results, open(results_file, "wb"), protocol=-1)


def _get_mean(results: Dict[str, Any], key: str) -> float:
    vals: List = [x for x, solved in zip(results[key], results["solved"]) if solved]
    if len(vals) == 0:
        return 0
    else:
        mean_val = np.mean([x for x, solved in zip(results[key], results["solved"]) if solved])
        return float(mean_val)


if __name__ == "__main__":
    main()
