"""
在选定的用户/LLM 子问题上，按单位流量=1 暴力枚举允许分流的方案，
并在该离散搜索空间内选出“服务率优先、成本次优”的 best 方案，用作对比基线。

注意：这里只在给定子问题和单位粒度下做穷举，不在全局意义上声称“最优解”。
"""

from __future__ import annotations

import copy
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import Entity
import analyze_bandwidth_llm_combined as ablc


def select_subproblem(users: Dict[int, Entity.User],
                      llms: Dict[int, Entity.LLM],
                      max_users: int,
                      max_llms: int) -> Tuple[List[int], List[int]]:
    """
    从全量 users/llms 中选取一个小子问题：
    - 用户按带宽从大到小排序，取前 max_users 个；
    - LLM 按 computation 从大到小排序，取前 max_llms 个。
    """
    sorted_users = sorted(users.items(),
                          key=lambda item: item[1].bw,
                          reverse=True)
    user_ids = [uid for uid, _ in sorted_users[:max_users]]

    sorted_llms = sorted(llms.items(),
                         key=lambda item: item[1].computation,
                         reverse=True)
    llm_ids = [lid for lid, _ in sorted_llms[:max_llms]]

    return user_ids, llm_ids


def run_bruteforce_split_bruteforce(
        network: Entity.Network,
        users: Dict[int, Entity.User],
        llms: Dict[int, Entity.LLM],
        user_ids: Sequence[int],
        llm_ids: Sequence[int],
) -> Tuple[List[Dict], Entity.Network, float, float, float, int]:
    """
    在给定子问题和单位粒度(1)下，暴力枚举允许分流的所有组合。

    前提：
        - 用户带宽 users[u].bw 为非负整数；
        - 单位流量为 1，每个单位可分配给任一 llm 或“不服务”；
        - 计算资源按 user.computation / user.bw 等比例分摊到单位流量。

    在该搜索空间内，用“服务率优先、成本次优”的规则选出一个 best 方案。

    返回:
        best_allocations: 当前搜索空间内 best 方案的 allocations 列表
        best_net: 对应的网络状态副本
        best_total_cost: best 方案的总开销
        best_service_rate: best 方案的服务率
        elapsed_time: 暴力搜索总耗时（秒）
        search_space_size: 实际枚举的方案数量
    """
    # 构造单位流量列表：unit_list[i] = user_id
    unit_list: List[int] = []
    ordered_user_ids: List[int] = list(user_ids)
    for uid in ordered_user_ids:
        bw_int = int(round(users[uid].bw))
        for _ in range(bw_int):
            unit_list.append(uid)

    total_demand_bw = float(len(unit_list))

    if total_demand_bw <= 0:
        empty_net = copy.deepcopy(network)
        return [], empty_net, 0.0, 0.0, 0.0, 1

    # 每个单位的决策：None（不服务）或某个 LLM
    choices: List[Optional[int]] = [None] + list(llm_ids)

    best_service_rate = -1.0
    best_total_cost = float("inf")
    best_allocations: List[Dict] = []
    best_net: Optional[Entity.Network] = None
    search_space_size = 0

    start_time = time.perf_counter()

    assignment: List[Optional[int]] = [None] * len(unit_list)

    def evaluate_current_assignment() -> None:
        nonlocal best_service_rate, best_total_cost, best_allocations, best_net, search_space_size

        search_space_size += 1

        net = copy.deepcopy(network)
        llms_copy: Dict[int, Entity.LLM] = copy.deepcopy(llms)

        allocations: List[Dict] = []
        total_cost = 0.0
        total_served_bw = 0.0

        for idx, uid in enumerate(unit_list):
            choice = assignment[idx]
            if choice is None:
                continue

            llm_id = choice
            user = users[uid]

            if user.bw <= 0:
                continue
            unit_cpu = float(user.computation) / float(user.bw)

            llm_obj = llms_copy.get(llm_id)
            if llm_obj is None:
                continue

            if llm_obj.available_computation < unit_cpu:
                continue

            distances, prev = net.dijkstra_with_capacity(uid,
                                                         min_capacity=1.0,
                                                         target_id=llm_id)
            if distances.get(llm_id, float("inf")) == float("inf"):
                continue

            node_path, link_path = net.get_path_with_links(prev, uid, llm_id)
            if not node_path:
                continue

            net.send_flow(link_path, 1.0)
            llm_obj.available_computation -= unit_cpu

            path_cost = float(distances[llm_id]) * 1.0
            total_cost += path_cost
            total_served_bw += 1.0

            allocations.append({
                "algorithm": "bruteforce-split",
                "user_id": uid,
                "llm_id": llm_id,
                "path": node_path,
                "cost": path_cost,
                "flow": 1.0,
            })

        if total_demand_bw > 0:
            service_rate = total_served_bw / total_demand_bw
        else:
            service_rate = 0.0

        # 先比较服务率，再比较总开销
        if service_rate > best_service_rate:
            best_service_rate = service_rate
            best_total_cost = total_cost
            best_allocations = allocations
            best_net = net
        elif service_rate == best_service_rate and total_cost < best_total_cost:
            best_total_cost = total_cost
            best_allocations = allocations
            best_net = net

    def dfs_unit(index: int) -> None:
        if index == len(unit_list):
            evaluate_current_assignment()
            return

        for choice in choices:
            assignment[index] = choice
            dfs_unit(index + 1)

    dfs_unit(0)

    elapsed_time = time.perf_counter() - start_time

    if best_net is None:
        best_net = copy.deepcopy(network)

    return best_allocations, best_net, best_total_cost, best_service_rate, elapsed_time, search_space_size


def compare_bruteforce_and_algorithms(
        user_distribution: str,
        llm_distribution: str,
        bandwidth: int,
        llm_computation: int,
        max_users: int,
        max_llms: int,
) -> pd.DataFrame:
    """
    对单个 (user_distribution, llm_distribution, bandwidth, llm_computation) 场景：
    - 在选定的 max_users×max_llms 子问题上运行暴力分流枚举；
    - 运行 no-split, 1-split, 1-split-augment, bottleneck-augment 四种现有算法；
    - 返回对比结果的 DataFrame。
    """
    json_data = Entity.load_network_from_sheets()
    network: Entity.Network = json_data["network"]
    llms_all: Dict[int, Entity.LLM] = Entity.load_llm_info(
        user_distribution, llm_distribution)
    users_all: Dict[int, Entity.User] = Entity.load_user_info(
        user_distribution)

    # 统一带宽和计算资源
    ablc.set_uniform_bandwidth(network, bandwidth)
    ablc.set_uniform_computation(llms_all, llm_computation)

    # 选择子问题
    user_ids, llm_ids = select_subproblem(users_all, llms_all, max_users,
                                          max_llms)
    users_sub = {uid: users_all[uid] for uid in user_ids}
    llms_sub = {lid: llms_all[lid] for lid in llm_ids}

    total_units = sum(int(round(u.bw)) for u in users_sub.values())

    records: List[Dict] = []

    # 暴力分流基线
    bf_allocs, _, bf_cost, bf_sr, bf_time, bf_space = run_bruteforce_split_bruteforce(
        network, users_sub, llms_sub, user_ids, llm_ids)

    records.append({
        "user_distribution": user_distribution,
        "llm_distribution": llm_distribution,
        "bandwidth": bandwidth,
        "llm_computation": llm_computation,
        "max_users": max_users,
        "max_llms": max_llms,
        "algorithm": "bruteforce-split",
        "total_cost": bf_cost,
        "service_rate": bf_sr,
        "runtime_sec": bf_time,
        "search_space_size": bf_space,
        "total_units": total_units,
        "num_users": len(user_ids),
        "num_llms": len(llm_ids),
        "num_allocations": len(bf_allocs),
    })

    # 现有四种算法
    algo_specs = [
        ("no-split", "no-split", None),
        ("1-split", "1-split", 1),
        ("1-split-augment", "1-split-augment", 1),
        ("bottleneck-augment", "bottleneck-augment", None),
    ]

    for label, algo_name, k in algo_specs:
        start = time.perf_counter()
        _, _, cost, sr = ablc.run_algorithm(network, users_sub, llms_sub,
                                            algo_name, k)
        elapsed = time.perf_counter() - start

        records.append({
            "user_distribution": user_distribution,
            "llm_distribution": llm_distribution,
            "bandwidth": bandwidth,
            "llm_computation": llm_computation,
            "max_users": max_users,
            "max_llms": max_llms,
            "algorithm": label,
            "total_cost": cost,
            "service_rate": sr,
            "runtime_sec": elapsed,
            "search_space_size": 0,
            "total_units": total_units,
            "num_users": len(user_ids),
            "num_llms": len(llm_ids),
            "num_allocations": None,
        })

    return pd.DataFrame.from_records(records)


def save_compare_results(df: pd.DataFrame, output_path: str) -> None:
    """
    将对比结果保存到指定 Excel 文件。
    若文件已存在，则追加新的 sheet（按 run_1, run_2, ... 命名）。
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_path):
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="run_1", index=False)
        return

    existing = pd.ExcelFile(output_path)
    existing_sheet_count = len(existing.sheet_names)
    new_sheet_name = f"run_{existing_sheet_count + 1}"

    with pd.ExcelWriter(output_path, engine="openpyxl",
                        mode="a") as writer:
        df.to_excel(writer, sheet_name=new_sheet_name, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bruteforce split baseline vs existing algorithms.")
    parser.add_argument("--user-distribution",
                        type=str,
                        default=Entity.DISTRIBUTION_TYPES[0],
                        help="用户分布名称，来自 Entity.DISTRIBUTION_TYPES")
    parser.add_argument("--llm-distribution",
                        type=str,
                        default=Entity.DISTRIBUTION_TYPES[0],
                        help="LLM 分布名称，来自 Entity.DISTRIBUTION_TYPES")
    parser.add_argument("--bandwidth",
                        type=int,
                        required=True,
                        help="统一链路带宽设置")
    parser.add_argument("--llm-computation",
                        type=int,
                        required=True,
                        help="统一 LLM 计算资源设置")
    parser.add_argument("--max-users",
                        type=int,
                        default=3,
                        help="暴力子问题参与的最大用户数")
    parser.add_argument("--max-llms",
                        type=int,
                        default=2,
                        help="暴力子问题参与的最大 LLM 数")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__),
                             "bruteforce_compare.xlsx"),
        help="结果输出 Excel 路径（默认 force/bruteforce_compare.xlsx）",
    )

    args = parser.parse_args()

    df = compare_bruteforce_and_algorithms(
        user_distribution=args.user_distribution,
        llm_distribution=args.llm_distribution,
        bandwidth=args.bandwidth,
        llm_computation=args.llm_computation,
        max_users=args.max_users,
        max_llms=args.max_llms,
    )

    save_compare_results(df, args.output)


if __name__ == "__main__":
    main()

