"""
实验主流程模块

根据ExperimentSetting.md实现完整的实验流程：
1. 多层循环：网络规模 -> 带宽 -> LLM容量 -> user分布 -> llm分布
2. 运行所有对比算法并计时
3. 计算性能指标：总开销、服务率、关键链路利用率、平均距离
4. 保存结果到Excel
"""

import os
import time
import copy
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import Entity
import Algorithm

# 实验配置
NETWORK_SIZES = [20]  # 中等规模网络节点数
NETWORK_BANDWIDTHS = [50, 60, 80, 100, 150]  # 全网络带宽（Gbps）- 恢复到合理范围
LLM_CAPACITIES_MEDIUM = [80, 100, 120]  # 中等规模LLM容量（Gbps）- 保持竞争性
LLM_CAPACITIES_LARGE = [200, 250, 300]  # 大规模LLM容量（Gbps）
DISTRIBUTIONS = ['uniform']  # 节点分布 - 先测试单一分布

# 算法列表（需要user_ideal_llms的算法单独标记）
ALGORITHMS = [
    {
        'name': 'no-split-aggregate',
        'need_ideal': False
    },
    {
        'name': 'no-split',
        'need_ideal': True
    },
    {
        'name': '1-split',
        'need_ideal': True
    },
    {
        'name': '1-split-no-aggregate',
        'need_ideal': True
    },
    {
        'name': 'bottleneck-split',
        'need_ideal': True
    },
    {
        'name': 'bottleneck-split-no-aggregate',
        'need_ideal': True
    },
    {
        'name': 'LLM-split-aggregate',
        'need_ideal': True
    },
    {
        'name': 'LLM-split',
        'need_ideal': True
    },
    {
        'name': '1-split-augment',
        'need_ideal': False
    },
    {
        'name': 'bottleneck-split-augment',
        'need_ideal': False
    },
]


def calculate_key_link_utilization(network, top_n=20) -> float:
    """
    计算前N条关键链路的平均利用率（根据介数中心性排列）

    Args:
        network: Network对象（运行算法后的状态）
        top_n: 前N条关键链路

    Returns:
        平均利用率（0-1之间）
    """
    import networkx as nx

    # 构建NetworkX图用于计算介数中心性
    G = nx.DiGraph()

    # 添加所有节点
    for node_id in network.nodes.keys():
        G.add_node(node_id)

    # 添加所有正向边（非反向边）
    edge_flows = {}
    edge_capacities = {}
    for src, links in network.links.items():
        for link in links:
            if not link.is_reverse:
                G.add_edge(link.src, link.dst)
                edge_key = (link.src, link.dst)
                edge_flows[edge_key] = link.flow
                edge_capacities[edge_key] = link.capacity

    # 计算边介数中心性
    try:
        edge_betweenness = nx.edge_betweenness_centrality(G, weight=None)
    except:
        # 如果图不连通或其他错误，返回0
        return 0.0

    # 按介数中心性排序
    sorted_edges = sorted(edge_betweenness.items(),
                          key=lambda x: x[1],
                          reverse=True)

    # 取前top_n条边
    top_edges = sorted_edges[:top_n]

    # 计算这些边的平均利用率
    utilizations = []
    for edge, _ in top_edges:
        if edge in edge_flows and edge in edge_capacities:
            capacity = edge_capacities[edge]
            if capacity > 1e-9:
                util = edge_flows[edge] / capacity
                utilizations.append(util)

    if utilizations:
        return np.mean(utilizations)
    else:
        return 0.0


def calculate_avg_distance_llm_to_user(network, users: Dict,
                                       llms: Dict) -> float:
    """
    计算user到LLM的平均距离（sigmoid归一化）

    Args:
        network: Network对象
        users: 用户字典
        llms: LLM字典

    Returns:
        归一化后的平均距离
    """
    distances = []

    for uid in users.keys():
        for lid in llms.keys():
            # 使用Dijkstra计算距离（从user到LLM，忽略容量约束）
            dist, _ = network.dijkstra_with_capacity(uid, 0, target_id=lid)
            if dist[lid] < float('inf'):
                distances.append(dist[lid])

    if distances:
        # 所有距离加和
        total_distance = sum(distances)
        # sigmoid归一化：1 / (1 + exp(-x/scale))
        scale = 100  # 调整scale以适应距离范围
        normalized = 1 / (1 + np.exp(-total_distance / scale))
        return normalized
    else:
        return 0.0


def load_network_data(network_size: int,
                      user_dist: str,
                      llm_dist: str,
                      bandwidth: int,
                      llm_capacity: int,
                      sheets_root='sheets'):
    """
    加载网络数据

    Args:
        network_size: 网络节点数（20, 40, 60, 80）
        user_dist: 用户分布
        llm_dist: LLM分布
        bandwidth: 网络带宽（Gbps）
        llm_capacity: LLM服务容量（Gbps）
        sheets_root: sheets根目录

    Returns:
        (network, users, llms, user_ideal_llms)
    """
    # 网络数据路径：sheets/N-xxx/
    network_dir = os.path.join(sheets_root, f'N-{network_size}')

    # 加载用户和LLM信息
    users = Entity.load_user_info(user_dist, sheets_root=network_dir)
    llms = Entity.load_llm_info(user_dist, llm_dist, sheets_root=network_dir)

    # 加载网络拓扑
    json_obj = Entity.load_network_from_sheets(llm_ids=llms.keys(),
                                               sheets_root=network_dir)
    network = json_obj['network']

    # 调整网络带宽和LLM容量
    # 1. 设置所有链路带宽为指定值
    for src, links in network.links.items():
        for link in links:
            if not link.is_reverse:
                link.capacity = bandwidth
                link.flow = 0  # 重置流量，residual_capacity会自动计算

    # 2. 设置所有LLM服务容量为指定值
    for llm in llms.values():
        llm.service_capacity = llm_capacity

    # 预计算user_ideal_llms（部分算法需要）
    user_ideal_llms = {}
    for uid, user in users.items():
        distances, _ = network.dijkstra_ideal(uid, user.bw)
        sorted_llms = sorted(distances.items(), key=lambda x: x[1])
        user_ideal_llms[uid] = {
            lid: dist
            for lid, dist in sorted_llms if lid in llms
        }

    return network, users, llms, user_ideal_llms


def run_single_experiment(network, users: Dict, llms: Dict,
                          user_ideal_llms: Dict, algorithm_name: str,
                          need_ideal: bool) -> Dict[str, Any]:
    """
    运行单次实验

    Args:
        network: Network对象
        users: 用户字典
        llms: LLM字典
        user_ideal_llms: 用户理想LLM字典
        algorithm_name: 算法名称
        need_ideal: 是否需要user_ideal_llms参数

    Returns:
        {
            'algorithm': str,
            'runtime': float,
            'total_cost': float,
            'acceptance_ratio': float,
            'key_link_util': float,
            'avg_distance': float
        }
    """
    # 深拷贝网络（避免修改原始网络）
    net_copy = copy.deepcopy(network)

    # 计时开始（包括BFS重建和消除负环的时间）
    start_time = time.time()

    # 运行算法（只运行一次）
    try:
        if need_ideal:
            result = Algorithm.run_algorithm(algorithm_name,
                                             net_copy,
                                             users,
                                             llms,
                                             user_ideal_llms=user_ideal_llms,
                                             is_shared=True)
        else:
            result = Algorithm.run_algorithm(algorithm_name,
                                             net_copy,
                                             users,
                                             llms,
                                             is_shared=True)
    except Exception as e:
        print(f"    算法 {algorithm_name} 运行失败: {e}")
        return None

    # 计时结束
    end_time = time.time()
    runtime = end_time - start_time

    # 提取基本指标
    total_cost = result['total_cost']
    acceptance_ratio = result['acceptance_ratio']

    # 计算关键链路利用率（使用算法返回的网络状态）
    result_network = result.get('network', net_copy)
    key_link_util = calculate_key_link_utilization(result_network, top_n=20)

    # 计算平均距离（使用原始网络）
    avg_distance = calculate_avg_distance_llm_to_user(network, users, llms)

    # 提取round_allocations（如果是augment算法）
    round_allocations = result.get('round_allocations', None)

    return {
        'algorithm': algorithm_name,
        'runtime': runtime,
        'total_cost': total_cost,
        'acceptance_ratio': acceptance_ratio,
        'key_link_util': key_link_util,
        'avg_distance': avg_distance,
        'round_allocations': round_allocations  # 用于withdraw-optimization记录
    }


def run_experiments_for_network_size(network_size: int,
                                     sheets_root='sheets',
                                     results_root='results'):
    """
    运行指定网络规模的所有实验

    Args:
        network_size: 网络节点数
        sheets_root: sheets根目录
        results_root: results根目录
    """
    print(f"\n{'='*80}")
    print(f"开始运行网络规模 N-{network_size} 的实验")
    print(f"{'='*80}")

    # 创建结果目录
    network_results_dir = os.path.join(results_root, f'N-{network_size}')
    os.makedirs(network_results_dir, exist_ok=True)

    # 选择LLM容量范围
    if network_size <= 80:
        llm_capacities = LLM_CAPACITIES_MEDIUM
    else:
        llm_capacities = LLM_CAPACITIES_LARGE

    # 存储详细结果（用于N-xxx-results.xlsx）
    detailed_results = []

    # 存储规模对比结果（用于N-size-results.xlsx）
    size_comparison_results = []

    # 存储withdraw-optimization结果（用于withdraw-optimization.xlsx）
    # 数据结构: {(bandwidth, llm_capacity, user_dist, llm_dist): {'1-split-no-aggregate': [...], '1-split': [...], '1-split-augment': [...]}}
    withdraw_optimization_data = {}

    # 多层循环
    total_experiments = len(NETWORK_BANDWIDTHS) * len(llm_capacities) * len(
        DISTRIBUTIONS) * len(DISTRIBUTIONS) * len(ALGORITHMS)
    current_experiment = 0

    for bandwidth in NETWORK_BANDWIDTHS:
        for llm_capacity in llm_capacities:
            for user_dist in DISTRIBUTIONS:
                for llm_dist in DISTRIBUTIONS:
                    print(
                        f"\n网络设置: 带宽={bandwidth}Gbps, LLM容量={llm_capacity}Gbps, "
                        f"user分布={user_dist}, llm分布={llm_dist}")

                    # 加载网络数据
                    try:
                        network, users, llms, user_ideal_llms = load_network_data(
                            network_size, user_dist, llm_dist, bandwidth,
                            llm_capacity, sheets_root)
                    except Exception as e:
                        print(f"  加载网络数据失败: {e}")
                        continue

                    # 配置标识
                    config_key = (bandwidth, llm_capacity, user_dist, llm_dist)
                    if config_key not in withdraw_optimization_data:
                        withdraw_optimization_data[config_key] = {}

                    # 运行所有算法
                    for algo_config in ALGORITHMS:
                        current_experiment += 1
                        algo_name = algo_config['name']
                        need_ideal = algo_config['need_ideal']

                        print(
                            f"  [{current_experiment}/{total_experiments}] 运行算法: {algo_name}"
                        )

                        # 运行实验
                        exp_result = run_single_experiment(
                            network, users, llms, user_ideal_llms, algo_name,
                            need_ideal)

                        if exp_result is None:
                            continue

                        # 记录详细结果（优化率后续在保存前统一补充）
                        detailed_results.append({
                            '带宽设置':
                            bandwidth,
                            'LLM服务容量设置':
                            llm_capacity,
                            'user分布':
                            user_dist,
                            'llm分布':
                            llm_dist,
                            '算法名':
                            algo_name,
                            '运行时间':
                            exp_result['runtime'],
                            '总开销':
                            exp_result['total_cost'],
                            '服务率':
                            exp_result['acceptance_ratio'],
                            '关键链路利用率':
                            exp_result['key_link_util'],
                            '平均距离':
                            exp_result['avg_distance']
                        })

                        # 记录规模对比结果
                        size_comparison_results.append({
                            'user_llm_dist':
                            f"{user_dist}-{llm_dist}",
                            '带宽设置':
                            bandwidth,
                            'LLM服务容量设置':
                            llm_capacity,
                            '网络节点个数':
                            network_size,
                            '算法名':
                            algo_name,
                            '运行时间':
                            exp_result['runtime'],
                            '总花销':
                            exp_result['total_cost'],
                            '服务率':
                            exp_result['acceptance_ratio']
                        })

                        # 记录withdraw-optimization结果（记录1-split-no-aggregate、1-split和1-split-augment）
                        if algo_name in [
                                '1-split-no-aggregate', '1-split',
                                '1-split-augment'
                        ] and exp_result['round_allocations'] is not None:
                            withdraw_optimization_data[config_key][
                                algo_name] = exp_result['round_allocations']

    # 保存详细结果到N-xxx-results.xlsx，增加“优化率”字段
    detailed_df = pd.DataFrame(detailed_results)

    if not detailed_df.empty:
        # 以同一网络配置（带宽、LLM容量、user分布、llm分布）分组
        group_cols = ['带宽设置', 'LLM服务容量设置', 'user分布', 'llm分布']

        def compute_optimization_rate(group: pd.DataFrame) -> pd.DataFrame:
            # 在该网络配置下找到1-split-augment的总开销作为基准
            mask_baseline = group['算法名'] == '1-split-augment'
            if not mask_baseline.any():
                group['优化率'] = np.nan
                return group

            baseline_cost = group.loc[mask_baseline, '总开销'].iloc[0]
            if baseline_cost == 0:
                group['优化率'] = np.nan
                return group

            group['优化率'] = (group['总开销'] - baseline_cost) / baseline_cost
            return group

        detailed_df = detailed_df.groupby(
            group_cols, as_index=False,
            group_keys=False).apply(compute_optimization_rate)

    detailed_excel_path = os.path.join(network_results_dir,
                                       f'N-{network_size}-results.xlsx')
    detailed_df.to_excel(detailed_excel_path, index=False, engine='openpyxl')
    print(f"\n详细结果已保存到: {detailed_excel_path}")

    # 保存规模对比结果到N-size-results.xlsx
    save_size_comparison_results(size_comparison_results, network_size,
                                 results_root)

    # 保存withdraw-optimization结果到withdraw-optimization.xlsx
    save_withdraw_optimization_results(withdraw_optimization_data,
                                       network_size, results_root)


def save_size_comparison_results(results: List[Dict],
                                 network_size: int,
                                 results_root='results'):
    """
    保存规模对比结果到N-size-results.xlsx

    Args:
        results: 结果列表
        network_size: 网络节点数
        results_root: results根目录
    """
    if not results:
        print(f"规模对比结果为空，跳过保存 N-{network_size}-size-results.xlsx")
        return

    # 按user_llm_dist分组
    grouped_results = {}
    for result in results:
        user_llm_dist = result.pop('user_llm_dist')
        if user_llm_dist not in grouped_results:
            grouped_results[user_llm_dist] = []
        grouped_results[user_llm_dist].append(result)

    if not grouped_results:
        print(f"规模对比结果分组为空，跳过保存 N-{network_size}-size-results.xlsx")
        return

    # 保存到Excel（每个sheet对应一个user_llm_dist）
    excel_path = os.path.join(results_root,
                              f'N-{network_size}-size-results.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for user_llm_dist, dist_results in grouped_results.items():
            df = pd.DataFrame(dist_results)
            # sheet名不能超过31个字符
            sheet_name = user_llm_dist[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"规模对比结果已保存到: {excel_path}")


def save_withdraw_optimization_results(data: Dict,
                                       network_size: int,
                                       results_root='results'):
    """
    保存withdraw-optimization结果到withdraw-optimization.xlsx

    新格式：每行包含三个算法在同一轮次的累积开销
    列：带宽设置、LLM服务容量设置、1-split-no-aggregate算法名、1-split算法名、1-split-augment算法名、
        迭代次数、1-split-no-aggregate截至当前轮次总花销、1-split截至当前轮次总花销、1-split-augment截至当前轮次总花销

    Args:
        data: {(bandwidth, llm_capacity, user_dist, llm_dist): {'1-split-no-aggregate': [...], '1-split': [...], '1-split-augment': [...]}}
        network_size: 网络节点数
        results_root: results根目录
    """
    if not data:
        print(f"无withdraw-optimization数据需要保存")
        return

    # 按user_llm_dist分组
    grouped_results = {}

    for config_key, algo_results in data.items():
        bandwidth, llm_capacity, user_dist, llm_dist = config_key
        user_llm_dist = f"{user_dist}-{llm_dist}"

        if user_llm_dist not in grouped_results:
            grouped_results[user_llm_dist] = []

        # 获取三个算法的round_allocations
        no_agg_rounds = algo_results.get('1-split-no-aggregate', [])
        one_split_rounds = algo_results.get('1-split', [])
        augment_rounds = algo_results.get('1-split-augment', [])

        # 找到最大轮次数
        max_rounds = max(
            len(no_agg_rounds) if no_agg_rounds else 0,
            len(one_split_rounds) if one_split_rounds else 0,
            len(augment_rounds) if augment_rounds else 0)

        # 合并同一轮次的数据
        for round_idx in range(max_rounds):
            row = {
                '带宽设置':
                bandwidth,
                'LLM服务容量设置':
                llm_capacity,
                '1-split-no-aggregate':
                '1-split-no-aggregate',
                '1-split':
                '1-split',
                '1-split-augment':
                '1-split-augment',
                '迭代次数':
                round_idx + 1,
                '1-split-no-aggregate截至当前轮次总花销':
                no_agg_rounds[round_idx]['cumulative_cost']
                if round_idx < len(no_agg_rounds) else np.nan,
                '1-split截至当前轮次总花销':
                one_split_rounds[round_idx]['cumulative_cost']
                if round_idx < len(one_split_rounds) else np.nan,
                '1-split-augment截至当前轮次总花销':
                augment_rounds[round_idx]['cumulative_cost']
                if round_idx < len(augment_rounds) else np.nan
            }
            grouped_results[user_llm_dist].append(row)

    if not grouped_results:
        print(f"withdraw-optimization数据分组为空，跳过保存")
        return

    # 保存到Excel（每个sheet对应一个user_llm_dist）
    network_results_dir = os.path.join(results_root, f'N-{network_size}')
    excel_path = os.path.join(network_results_dir,
                              'withdraw-optimization.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for user_llm_dist, dist_results in grouped_results.items():
            df = pd.DataFrame(dist_results)
            # sheet名不能超过31个字符
            sheet_name = user_llm_dist[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"withdraw-optimization结果已保存到: {excel_path}")


def run_all_experiments(sheets_root='sheets', results_root='results'):
    """
    运行所有实验

    Args:
        sheets_root: sheets根目录
        results_root: results根目录
    """
    # 创建results目录
    os.makedirs(results_root, exist_ok=True)

    # 对每个网络规模运行实验
    for network_size in NETWORK_SIZES:
        try:
            run_experiments_for_network_size(network_size, sheets_root,
                                             results_root)
        except Exception as e:
            print(f"\n网络规模 N-{network_size} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("所有实验完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    # 运行所有实验
    run_all_experiments()
