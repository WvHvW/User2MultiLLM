from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from force import brute_force_flow_unit_comparison as bf


RANDOM_SEED: int = 2025
NUM_INSTANCES_DEFAULT: int = 30
TOTAL_DEMAND_MIN: int = 8
TOTAL_DEMAND_MAX: int = 10
CAPACITY_RATIO_MIN: float = 0.6
CAPACITY_RATIO_MAX: float = 1.0
COST_MIN: float = 0.5
COST_MAX: float = 3.0

UserDemands = Dict[int, int]
LlmCapacities = Dict[int, int]
PairCosts = Dict[Tuple[int, int], float]


@dataclass
class InstanceConfig:
    instance_id: int
    user_demands: UserDemands
    llm_capacities: LlmCapacities
    pair_costs: PairCosts
    total_demand_units: int
    assignment_space_size: int


def _random_positive_partition(rng: random.Random,
                               total: int,
                               num_parts: int) -> List[int]:
    if total < num_parts or num_parts <= 0:
        raise ValueError("total must be >= num_parts and num_parts > 0")
    if num_parts == 1:
        return [total]

    cut_points = sorted(
        rng.sample(range(1, total), k=num_parts - 1))  # type: ignore[arg-type]
    parts: List[int] = []
    previous = 0
    for cut in cut_points + [total]:
        parts.append(cut - previous)
        previous = cut
    return parts


def generate_random_instance(instance_id: int,
                             rng: random.Random) -> InstanceConfig:
    user_ids: List[int] = list(bf.USER_NODE_IDS)
    llm_ids: List[int] = list(bf.LLM_NODE_IDS)

    num_users = len(user_ids)
    num_llms = len(llm_ids)

    total_demand_units = rng.randint(TOTAL_DEMAND_MIN, TOTAL_DEMAND_MAX)

    user_parts = _random_positive_partition(rng, total_demand_units,
                                            num_users)
    user_demands: UserDemands = {
        user_id: demand
        for user_id, demand in zip(user_ids, user_parts)
    }

    min_capacity_total = max(num_llms,
                             int(total_demand_units * CAPACITY_RATIO_MIN))
    max_capacity_total = max(min_capacity_total,
                             int(total_demand_units * CAPACITY_RATIO_MAX))
    max_capacity_total = min(max_capacity_total, total_demand_units)

    capacity_total = rng.randint(min_capacity_total, max_capacity_total)
    llm_parts = _random_positive_partition(rng, capacity_total, num_llms)
    llm_capacities: LlmCapacities = {
        llm_id: cap
        for llm_id, cap in zip(llm_ids, llm_parts)
    }

    pair_costs: PairCosts = {}
    for user_id in user_ids:
        for llm_id in llm_ids:
            cost = rng.uniform(COST_MIN, COST_MAX)
            pair_costs[(user_id, llm_id)] = float(cost)

    assignment_space_size = (num_llms + 1)**total_demand_units

    return InstanceConfig(instance_id=instance_id,
                          user_demands=user_demands,
                          llm_capacities=llm_capacities,
                          pair_costs=pair_costs,
                          total_demand_units=total_demand_units,
                          assignment_space_size=assignment_space_size)


def _set_problem_in_bf_module(config: InstanceConfig) -> None:
    bf.USER_DEMAND_UNITS = dict(config.user_demands)
    bf.LLM_CAPACITY_UNITS = dict(config.llm_capacities)
    bf.PAIR_COSTS = dict(config.pair_costs)
    bf.TOTAL_DEMAND_UNITS = int(config.total_demand_units)
    num_llm = len(bf.LLM_NODE_IDS)
    bf.ASSIGNMENT_SEARCH_SPACE_SIZE = (num_llm + 1)**bf.TOTAL_DEMAND_UNITS


def evaluate_single_instance(config: InstanceConfig) -> List[Dict[str, Any]]:
    _set_problem_in_bf_module(config)

    brute_result = bf.brute_force_unit_assignment()
    one_split_result = bf.run_one_split_augment()
    bottleneck_result = bf.run_bottleneck_augment()

    total_demand = float(config.total_demand_units)
    if total_demand <= 0:
        raise ValueError("total_demand_units must be positive.")

    def served_units(result: bf.AlgorithmResult) -> float:
        return result.service_rate * total_demand

    brute_served = served_units(brute_result)
    if brute_served <= 0:
        raise ValueError("Bruteforce solution served no flow.")
    brute_unit_cost = brute_result.total_cost / brute_served
    brute_service_rate = brute_result.service_rate
    brute_time = brute_result.elapsed_seconds

    results_rows: List[Dict[str, Any]] = []

    def add_row(result: bf.AlgorithmResult) -> None:
        served = served_units(result)
        unit_cost = result.total_cost / served if served > 0 else float("inf")
        if brute_service_rate > 0:
            approx_service_ratio = result.service_rate / brute_service_rate
        else:
            approx_service_ratio = 1.0
        if brute_unit_cost > 0 and unit_cost != float("inf"):
            approx_unit_cost_ratio = unit_cost / brute_unit_cost
        else:
            approx_unit_cost_ratio = 1.0
        if brute_time > 0:
            time_ratio_vs_brute = result.elapsed_seconds / brute_time
        else:
            time_ratio_vs_brute = 1.0

        results_rows.append({
            "instance_id": config.instance_id,
            "algorithm": result.name,
            "total_demand_units": config.total_demand_units,
            "assignment_space_size": config.assignment_space_size,
            "service_rate": result.service_rate,
            "total_cost": result.total_cost,
            "elapsed_seconds": result.elapsed_seconds,
            "served_units": served,
            "unit_cost": unit_cost,
            "approx_service_ratio": approx_service_ratio,
            "approx_unit_cost_ratio": approx_unit_cost_ratio,
            "time_ratio_vs_bruteforce": time_ratio_vs_brute,
        })

    add_row(brute_result)
    add_row(one_split_result)
    add_row(bottleneck_result)

    return results_rows


def run_experiment(num_instances: int = NUM_INSTANCES_DEFAULT,
                   seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    all_rows: List[Dict[str, Any]] = []

    for instance_id in range(num_instances):
        config = generate_random_instance(instance_id, rng)
        rows = evaluate_single_instance(config)
        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)

    summary_rows: List[Dict[str, Any]] = []
    for algorithm_name, group in raw_df.groupby("algorithm"):
        service_ratios = group["approx_service_ratio"]
        unit_cost_ratios = group["approx_unit_cost_ratio"]
        time_ratios = group["time_ratio_vs_bruteforce"]

        summary_rows.append({
            "algorithm": algorithm_name,
            "num_instances": group["instance_id"].nunique(),
            "service_ratio_min": float(service_ratios.min()),
            "service_ratio_mean": float(service_ratios.mean()),
            "unit_cost_ratio_max": float(unit_cost_ratios.max()),
            "unit_cost_ratio_mean": float(unit_cost_ratios.mean()),
            "time_ratio_min": float(time_ratios.min()),
            "time_ratio_mean": float(time_ratios.mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("algorithm").reset_index(drop=True)

    return raw_df, summary_df


def save_results_to_excel(raw_df: pd.DataFrame,
                          summary_df: pd.DataFrame,
                          output_path: str) -> None:
    results_dir = os.path.dirname(output_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="instances", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)


def main() -> None:
    raw_df, summary_df = run_experiment()
    output_path = os.path.join(CURRENT_DIR, "results",
                               "brute_force_approx_ratio.xlsx")
    save_results_to_excel(raw_df, summary_df, output_path)


if __name__ == "__main__":
    main()

