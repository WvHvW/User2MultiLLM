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
NETWORK_SIZES = [20, 40, 60, 80, 100, 200]
NETWORK_BANDWIDTHS = [50, 75, 100, 200, 400]  # 全网络带宽（Gbps）
LLM_CAPACITIES_MEDIUM = [80, 100, 150, 200]  # 中等规模LLM容量（Gbps）
LLM_CAPACITIES_LARGE = [250, 300, 400, 500]  # 大规模LLM容量（Gbps）
DISTRIBUTIONS = ['uniform', 'poisson', 'sparse', 'gaussian']

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
    {
        'name': 'NW-1-split-augment',
        'need_ideal': False
    },
    {
        'name': 'NW-bottleneck-split-augment',
        'need_ideal': False
    },
]


def calculate_key_link_utilization(network,
                                   users: Dict,
                                   llms: Dict,
                                   top_n=20) -> float:
    """
    计算前N条关键链路的平均利用率（根据介数中心性排列）

    注意：介数中心性只计算User集合到LLM集合之间的路径

    Args:
        network: Network对象（运行算法后的状态）
        users: 用户字典
        llms: LLM字典
        top_n: 前N条关键链路

    Returns:
        平均利用率（0-1之间）
    """
    import networkx as nx

    # 构建NetworkX图用于计算介数中心性
    G = nx.Graph()  # 使用无向图（因为物理链路是双向的）

    # 定义超级源汇点（需要排除）
    S = -1
    T = -2
    virtual_nodes = {S, T}

    # 添加所有物理节点（排除S和T）
    edge_flows = {}
    edge_capacities = {}

    for node_id in network.nodes.keys():
        if node_id not in virtual_nodes:
            G.add_node(node_id)

    # 添加所有物理边（排除涉及S和T的边）
    for src, links in network.links.items():
        for link in links:
            if not link.is_reverse:
                # 排除涉及虚拟节点的边
                if link.src in virtual_nodes or link.dst in virtual_nodes:
                    continue

                # 无向图中只添加一次
                if not G.has_edge(link.src, link.dst):
                    G.add_edge(link.src, link.dst)
                edge_key = (link.src, link.dst)
                edge_flows[edge_key] = link.flow
                edge_capacities[edge_key] = link.capacity

    # 计算User集合到LLM集合之间的边介数中心性
    user_set = set(users.keys())
    llm_set = set(llms.keys())

    try:
        # 只计算从user集合到llm集合的最短路径经过的边
        edge_betweenness = nx.edge_betweenness_centrality_subset(
            G, sources=user_set, targets=llm_set, normalized=True)
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
        # 检查正向和反向边
        if edge in edge_flows and edge in edge_capacities:
            capacity = edge_capacities[edge]
            if capacity > 1e-9:
                util = edge_flows[edge] / capacity
                utilizations.append(util)
        elif (edge[1], edge[0]) in edge_flows and (edge[1],
                                                   edge[0]) in edge_capacities:
            # 检查反向边
            capacity = edge_capacities[(edge[1], edge[0])]
            if capacity > 1e-9:
                util = edge_flows[(edge[1], edge[0])] / capacity
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
    # 使用perf_counter提供纳秒级精度
    start_time = time.perf_counter()

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
    end_time = time.perf_counter()
    runtime = end_time - start_time

    # 提取基本指标
    total_cost = result['total_cost']
    acceptance_ratio = result['acceptance_ratio']

    # 计算关键链路利用率（使用算法返回的网络状态）
    result_network = result.get('network', net_copy)
    key_link_util = calculate_key_link_utilization(result_network,
                                                   users,
                                                   llms,
                                                   top_n=20)

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
    运行指定网络规模的所有实验（增量写入版本）

    关键改进：
    1. 每完成一个配置就立即写入Excel，避免内存累积
    2. 使用追加模式，即使中途失败也能保留已完成的数据
    3. 峰值内存从1.4GB降至约5MB

    Args:
        network_size: 网络节点数
        sheets_root: sheets根目录
        results_root: results根目录
    """
    print(f"\n{'='*80}")
    print(f"开始运行网络规模 N-{network_size} 的实验（增量写入模式）")
    print(f"{'='*80}")

    # 创建结果目录
    network_results_dir = os.path.join(results_root, f'N-{network_size}')
    os.makedirs(network_results_dir, exist_ok=True)

    # 选择LLM容量范围
    if network_size <= 80:
        llm_capacities = LLM_CAPACITIES_MEDIUM
    else:
        llm_capacities = LLM_CAPACITIES_LARGE

    # 初始化Excel文件路径
    detailed_excel_path = os.path.join(network_results_dir,
                                       f'N-{network_size}-results.xlsx')
    size_excel_path = os.path.join(results_root,
                                   f'N-{network_size}-size-results.xlsx')
    withdraw_excel_path = os.path.join(network_results_dir,
                                       'withdraw-optimization.xlsx')

    # 删除旧文件（如果存在）
    for path in [detailed_excel_path, size_excel_path, withdraw_excel_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"删除旧文件: {path}")

    # 统计信息
    total_configs = len(NETWORK_BANDWIDTHS) * len(llm_capacities) * len(
        DISTRIBUTIONS) * len(DISTRIBUTIONS)
    total_experiments = total_configs * len(ALGORITHMS)
    current_config = 0
    current_experiment = 0

    # 外层循环：每个配置
    for bandwidth in NETWORK_BANDWIDTHS:
        for llm_capacity in llm_capacities:
            for user_dist in DISTRIBUTIONS:
                for llm_dist in DISTRIBUTIONS:
                    current_config += 1
                    print(
                        f"\n[配置 {current_config}/{total_configs}] 带宽={bandwidth}Gbps, "
                        f"LLM容量={llm_capacity}Gbps, user={user_dist}, llm={llm_dist}"
                    )

                    # 加载网络数据
                    try:
                        network, users, llms, user_ideal_llms = load_network_data(
                            network_size, user_dist, llm_dist, bandwidth,
                            llm_capacity, sheets_root)
                    except Exception as e:
                        print(f"  [错误] 加载网络数据失败: {e}")
                        continue

                    # 临时存储当前配置的结果（内存占用小，只有12个算法的结果）
                    config_detailed_results = []
                    config_size_results = []
                    config_withdraw_data = {}

                    # 运行所有算法
                    for algo_config in ALGORITHMS:
                        current_experiment += 1
                        algo_name = algo_config['name']
                        need_ideal = algo_config['need_ideal']

                        print(
                            f"  [{current_experiment}/{total_experiments}] {algo_name}"
                        )

                        # 运行实验
                        exp_result = run_single_experiment(
                            network, users, llms, user_ideal_llms, algo_name,
                            need_ideal)

                        if exp_result is None:
                            continue

                        # 记录详细结果
                        config_detailed_results.append({
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
                        config_size_results.append({
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

                        # 记录withdraw-optimization结果
                        if algo_name in [
                                '1-split-no-aggregate', '1-split',
                                '1-split-augment'
                        ] and exp_result['round_allocations'] is not None:
                            config_withdraw_data[algo_name] = exp_result[
                                'round_allocations']

                    # 立即写入当前配置的结果（增量写入，避免内存累积）
                    if config_detailed_results:
                        append_detailed_results(detailed_excel_path,
                                                config_detailed_results)

                    if config_size_results:
                        append_size_results(size_excel_path,
                                            config_size_results)

                    if config_withdraw_data:
                        append_withdraw_results(withdraw_excel_path, bandwidth,
                                                llm_capacity, user_dist,
                                                llm_dist, config_withdraw_data)

                    print(f"  [完成] 配置结果已保存")

    print(f"\n{'='*80}")
    print(f"N-{network_size} 实验全部完成")
    print(f"详细结果: {detailed_excel_path}")
    print(f"规模对比: {size_excel_path}")
    print(f"Withdraw优化: {withdraw_excel_path}")
    print(f"{'='*80}")


def append_detailed_results(excel_path: str, config_results: List[Dict]):
    """
    追加详细结果到N-xxx-results.xlsx

    Args:
        excel_path: Excel文件路径
        config_results: 当前配置的结果列表（12个算法）
    """
    df_new = pd.DataFrame(config_results)

    # 计算优化率（相对于1-split-augment基线）
    mask_baseline = df_new['算法名'] == '1-split-augment'
    if mask_baseline.any():
        baseline_cost = df_new.loc[mask_baseline, '总开销'].iloc[0]
        if baseline_cost > 1e-9:
            df_new['优化率'] = (df_new['总开销'] - baseline_cost) / baseline_cost
        else:
            df_new['优化率'] = np.nan
    else:
        df_new['优化率'] = np.nan

    # 追加写入
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path, engine='openpyxl')
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(excel_path, index=False, engine='openpyxl')
    else:
        df_new.to_excel(excel_path, index=False, engine='openpyxl')


def append_size_results(excel_path: str, config_results: List[Dict]):
    """
    追加规模对比结果到N-size-results.xlsx（按user_llm_dist分sheet）

    Args:
        excel_path: Excel文件路径
        config_results: 当前配置的结果列表
    """
    # 按user_llm_dist分组
    grouped = {}
    for data in config_results:
        user_llm_dist = data.pop('user_llm_dist')
        sheet_name = user_llm_dist[:31]  # sheet名不能超过31字符
        if sheet_name not in grouped:
            grouped[sheet_name] = []
        grouped[sheet_name].append(data)

    # 追加写入
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path,
                            engine='openpyxl',
                            mode='a',
                            if_sheet_exists='overlay') as writer:
            for sheet_name, sheet_data in grouped.items():
                df_new = pd.DataFrame(sheet_data)
                if sheet_name in writer.book.sheetnames:
                    # 读取现有数据并合并
                    df_existing = pd.read_excel(excel_path,
                                                sheet_name=sheet_name,
                                                engine='openpyxl')
                    df_combined = pd.concat([df_existing, df_new],
                                            ignore_index=True)
                    # 删除旧sheet并写入合并后的数据
                    del writer.book[sheet_name]
                    df_combined.to_excel(writer,
                                         sheet_name=sheet_name,
                                         index=False)
                else:
                    df_new.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 首次创建
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, sheet_data in grouped.items():
                df = pd.DataFrame(sheet_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def append_withdraw_results(excel_path: str, bandwidth: int, llm_capacity: int,
                            user_dist: str, llm_dist: str,
                            config_withdraw_data: Dict):
    """
    追加withdraw-optimization结果到withdraw-optimization.xlsx

    格式：每行包含三个算法在同一轮次的累积开销

    Args:
        excel_path: Excel文件路径
        bandwidth: 带宽设置
        llm_capacity: LLM服务容量设置
        user_dist: 用户分布
        llm_dist: LLM分布
        config_withdraw_data: {algo_name: round_allocations}
    """
    # 生成当前配置的数据行
    no_agg_rounds = config_withdraw_data.get('1-split-no-aggregate', [])
    one_split_rounds = config_withdraw_data.get('1-split', [])
    augment_rounds = config_withdraw_data.get('1-split-augment', [])

    max_rounds = max(
        len(no_agg_rounds) if no_agg_rounds else 0,
        len(one_split_rounds) if one_split_rounds else 0,
        len(augment_rounds) if augment_rounds else 0)

    rows = []
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
        rows.append(row)

    if not rows:
        return

    # 按user_llm_dist分sheet
    sheet_name = f"{user_dist}-{llm_dist}"[:31]
    df_new = pd.DataFrame(rows)

    # 追加写入
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path,
                            engine='openpyxl',
                            mode='a',
                            if_sheet_exists='overlay') as writer:
            if sheet_name in writer.book.sheetnames:
                # 读取现有数据并合并
                df_existing = pd.read_excel(excel_path,
                                            sheet_name=sheet_name,
                                            engine='openpyxl')
                df_combined = pd.concat([df_existing, df_new],
                                        ignore_index=True)
                # 删除旧sheet并写入合并后的数据
                del writer.book[sheet_name]
                df_combined.to_excel(writer,
                                     sheet_name=sheet_name,
                                     index=False)
            else:
                df_new.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 首次创建
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_new.to_excel(writer, sheet_name=sheet_name, index=False)


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
