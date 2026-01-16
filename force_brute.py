"""
真正的暴力枚举算法 (True Brute Force Baseline)

完整搜索空间：n! × M^n
- 枚举所有流量单位的路由顺序（n!种排列）
- 对每种顺序，枚举每个单位选择哪个LLM（M^n种分配）
- 按顺序逐个路由每个流量单位，考虑链路容量和LLM容量

剪枝优化：
1. 分支定界：当前cost超过最优解时停止
2. LLM容量剪枝：容量不足时停止当前方案
3. 链路可达性剪枝：无可达路径时停止当前方案

实现方式：迭代式（避免递归爆栈）

推荐配置：
- 总流量 ≤ 10单位（10! × 2^10 ≈ 37亿次搜索）
- 总流量 = 8单位（8! × 2^8 ≈ 1千万次搜索，几分钟）
"""

import os
import sys
import copy
import itertools
from typing import Dict, List, Tuple
from collections import defaultdict

from Entity import load_network_from_sheets, load_user_info, load_llm_info, User, LLM, Network


def brute_force_optimal(
        network: Network, users: Dict[int, User],
        llms: Dict[int, LLM]) -> Tuple[float, List, int, float]:
    """
    真正的暴力枚举最优解（迭代式实现）

    Args:
        network: 网络对象
        users: 用户字典
        llms: LLM字典

    Returns:
        (最优开销, 最优解, 实际搜索空间, 服务流量)

    算法：
        外层循环：枚举所有n!种流量单位的排列
        内层循环：对每种排列，枚举M^n种LLM分配
        路由过程：按排列顺序逐个路由每个流量单位

    注意：即使LLM容量不足，也会返回部分服务的最优解
    """
    from math import factorial

    # 1. 构建流量单位列表
    flow_units = []  # [user_id, user_id, ...] 每个元素代表1单位流量
    for user_id in sorted(users.keys()):
        demand = int(users[user_id].bw)
        for _ in range(demand):
            flow_units.append(user_id)

    n = len(flow_units)
    llm_list = sorted(llms.keys())
    M = len(llm_list)

    print(f"真正的暴力枚举算法（迭代式）:")
    print(f"  流量单位数: {n}")
    print(f"  流量单位详情: {flow_units}")
    print(f"  LLM数: {M}")
    print(f"  LLM容量: {[int(llms[lid].service_capacity) for lid in llm_list]}")
    print(f"  理论搜索空间: {n}! × {M}^{n} = {factorial(n):,} × {M**n:,}")
    print(f"                = {factorial(n) * (M ** n):,}")
    print(f"  优化: 分支定界 + 链路容量剪枝")
    print()

    # 搜索状态
    best_cost = float('inf')
    best_solution = None
    best_served_flow = 0  # 最优解的服务流量
    actual_search_space = 0
    pruned_by_cost = 0
    pruned_by_link_capacity = 0
    skipped_by_llm_capacity = 0  # 因LLM容量不足跳过的流量单位

    total_orderings = factorial(n)
    total_allocations_per_ordering = M**n

    print("开始搜索...")

    # 2. 外层循环：枚举所有n!种排列
    for ordering_idx, ordering in enumerate(itertools.permutations(range(n))):

        # 进度显示（每1000种排列）
        if (ordering_idx + 1) % 1000 == 0:
            print(f"  排列进度: {ordering_idx+1:,} / {total_orderings:,} "
                  f"({(ordering_idx+1)/total_orderings*100:.1f}%), "
                  f"搜索空间: {actual_search_space:,}, 当前最优: {best_cost:.2f}")

        # 3. 内层循环：对当前排列，枚举M^n种LLM分配
        for allocation in itertools.product(llm_list, repeat=n):
            # allocation[i] = 第i个流量单位分配给哪个LLM

            # 初始化状态
            net = copy.deepcopy(network)
            llms_remaining = {
                lid: llms[lid].service_capacity
                for lid in llm_list
            }
            current_cost = 0.0
            current_served = 0  # 当前方案实际服务的流量

            # 4. 按排列顺序逐个路由每个流量单位
            for order_pos in ordering:
                # order_pos = 当前要路由的流量单位索引
                user_id = flow_units[order_pos]
                llm_id = allocation[order_pos]

                # 检查LLM容量（不足时跳过该流量单位，继续尝试后续单位）
                if llms_remaining[llm_id] < 1 - 1e-6:
                    skipped_by_llm_capacity += 1
                    continue  # 跳过而不是break

                # 使用Dijkstra找最短路径（考虑残余容量）
                distances, previous_nodes = net.dijkstra_with_capacity(
                    start_node_id=user_id,
                    min_capacity=1,  # 需要至少1单位容量
                    target_id=llm_id)

                # 检查链路可达性（无路径时跳过）
                if distances[llm_id] == float('inf'):
                    pruned_by_link_capacity += 1
                    continue  # 跳过而不是break

                # 获取路径
                path_nodes, path_links = net.get_path_with_links(
                    previous_nodes, user_id, llm_id)

                # 沿路径发送1单位流量（直接修改flow属性）
                for link in path_links:
                    link.flow += 1

                # 累加开销和服务流量
                current_cost += distances[llm_id]
                current_served += 1
                llms_remaining[llm_id] -= 1

                # 剪枝: 分支定界（仅当服务流量相同时比较开销）
                # 但仍允许继续尝试后续流量单位，以便获得更高服务率
                if current_served == best_served_flow and current_cost >= best_cost:
                    pruned_by_cost += 1
                    continue

            # 计入实际搜索空间
            actual_search_space += 1

            # 更新最优解（优先选择服务流量更多的，其次选择开销更小的）
            if (current_served > best_served_flow
                    or (current_served == best_served_flow
                        and current_cost < best_cost)):
                best_cost = current_cost
                best_served_flow = current_served
                best_solution = (ordering, allocation)

    print(f"\n搜索完成:")
    print(f"  排列总数: {total_orderings:,}")
    print(f"  理论搜索空间: {total_orderings * total_allocations_per_ordering:,}")
    print(f"  实际搜索空间: {actual_search_space:,}")
    print(f"  分支定界剪枝: {pruned_by_cost:,}")
    print(f"  LLM容量跳过: {skipped_by_llm_capacity:,}")
    print(f"  链路容量剪枝: {pruned_by_link_capacity:,}")
    print(f"  最优开销: {best_cost:.2f}")
    print(f"  服务流量: {best_served_flow}/{n} ({best_served_flow/n*100:.1f}%)")

    return best_cost, best_solution, actual_search_space, best_served_flow


def format_solution(flow_units: List[int], solution: Tuple,
                    users: Dict[int, User]) -> Dict[int, Dict[int, int]]:
    """
    格式化解，统计每个用户分配给各LLM的流量

    Args:
        flow_units: 流量单位列表
        solution: (ordering, allocation)
        users: 用户字典

    Returns:
        {user_id: {llm_id: flow_amount}}
    """
    result = defaultdict(lambda: defaultdict(int))

    if solution:
        ordering, allocation = solution
        for order_pos in ordering:
            user_id = flow_units[order_pos]
            llm_id = allocation[order_pos]
            result[user_id][llm_id] += 1

    return dict(result)


def main():
    """主函数"""

    # 加载网络
    print("加载网络数据...")
    sheets_root = 'sheets/N-opt'

    json_obj = load_network_from_sheets(sheets_root=sheets_root)
    network = json_obj['network']
    nodes = json_obj['nodes']

    print(f"  网络节点数: {len(nodes)}")
    physical_edges = sum(1 for links in network.links.values()
                         for link in links if not link.is_reverse)
    print(f"  网络边数: {physical_edges}")
    print()

    # 加载用户和LLM
    print("加载用户和LLM...")
    users = load_user_info(distribution='uniform', sheets_root=sheets_root)
    llms = load_llm_info(user_distribution='uniform',
                         llm_distribution='uniform',
                         sheets_root=sheets_root)

    print(f"  用户数: {len(users)}")
    for user_id in sorted(users.keys()):
        print(f"    用户{user_id}: 需求{int(users[user_id].bw)}单位")

    print(f"  LLM数: {len(llms)}")
    for llm_id in sorted(llms.keys()):
        print(f"    LLM{llm_id}: 容量{int(llms[llm_id].service_capacity)}单位")
    print()

    # 警告检查
    total_flow = sum(int(u.bw) for u in users.values())
    if total_flow > 10:
        print(f"⚠️  警告: 总流量{total_flow}单位过大，搜索空间爆炸！")
        print(f"   建议降低到≤10单位，当前可能需要几小时到几天")
        response = input("   是否继续？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return

    # 运行搜索
    print()
    best_cost, best_solution, search_space = brute_force_optimal(
        network, users, llms)

    # 输出结果
    print()
    print("=" * 80)
    print("最优解:")
    print("=" * 80)

    flow_units = []
    for user_id in sorted(users.keys()):
        for _ in range(int(users[user_id].bw)):
            flow_units.append(user_id)

    allocation_summary = format_solution(flow_units, best_solution, users)

    for user_id in sorted(allocation_summary.keys()):
        print(f"用户{user_id} (总需求{int(users[user_id].bw)}单位):")
        for llm_id in sorted(allocation_summary[user_id].keys()):
            flow = allocation_summary[user_id][llm_id]
            print(f"  → LLM{llm_id}: {flow}单位")

    print()
    print(f"最优开销: {best_cost:.2f}")
    print(f"实际搜索空间: {search_space:,}")
    print("=" * 80)


if __name__ == '__main__':
    main()
