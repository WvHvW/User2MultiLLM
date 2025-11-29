"""
分析带宽和LLM计算资源组合影响

两个大循环：
1. 带宽范围（500-4000，步长250）嵌套LLM服务容量范围（8-32，步长2）
2. LLM服务容量范围（8-32，步长2）嵌套带宽范围（500-4000，步长250）

每个X对应两个Y轴：
- 左Y轴：optimization（相对于1-split-augment的成本百分比）
- 右Y轴：服务率（已分配流量/总需求）
"""

import Entity
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy

# 配置
DISTRIBUTION_TYPES = ["gaussian"]
is_shared = 1
BANDWIDTH_VALUES = list(range(500, 4500, 500))  # 500到4000，步长500
COMPUTATION_VALUES = list(range(8, 34, 4))  # 8到32，步长4
ALGORITHMS = ['no-split', '1-split', '1-split-augment', 'bottleneck-augment']
USER_COMPUTATION_PATTERNS = [
    [4, 4, 4, 4, 4, 4, 4, 4],
    # [6, 4, 4, 4, 4, 4, 4, 2],
    # [6, 6, 4, 4, 4, 4, 2, 2],
    # [6, 6, 6, 4, 4, 2, 2, 2],
    # [6, 6, 6, 6, 2, 2, 2, 2],
]
USER_COMPUTATION_TO_BANDWIDTH = {
    6: 750,
    4: 500,
    2: 250,
}

# 输出目录
_USER_PATTERN_DIR = os.path.join(os.path.dirname(__file__), 'userpattern')
os.makedirs(_USER_PATTERN_DIR, exist_ok=True)

print(f"输出目录已创建: {_USER_PATTERN_DIR}")


def compute_edge_betweenness(alo_network, users, llms, top_n=10):
    """计算边介数中心性"""
    import networkx as nx

    G = nx.DiGraph()
    for nid in alo_network.nodes:
        G.add_node(nid)

    for src, links in alo_network.links.items():
        for link in links:
            if not link.is_reverse:
                G.add_edge(src, link.dst, weight=link.distance)

    edge_betweenness = {}
    for user_id in users.keys():
        for llm_id in llms.keys():
            try:
                path = nx.shortest_path(G,
                                        source=user_id,
                                        target=llm_id,
                                        weight='weight')
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    edge_betweenness[edge] = edge_betweenness.get(edge, 0) + 1
            except nx.NetworkXNoPath:
                continue

    sorted_edges = sorted(edge_betweenness.items(),
                          key=lambda x: x[1],
                          reverse=True)
    return sorted_edges[:top_n]


def calculate_link_utilization(network, target_edges):
    """计算指定边的利用率"""
    edge_lookup = {}
    for src, links in network.links.items():
        for link in links:
            if link.is_reverse:
                continue
            edge_lookup[(link.src, link.dst)] = link

    utilization = {}
    for edge in target_edges:
        link = edge_lookup.get(edge)
        if not link or link.capacity <= 0:
            utilization[edge] = 0.0
        else:
            utilization[edge] = link.flow / link.capacity
    return utilization


def compute_llm_user_distance_indicator(network, users, llms):
    """
    计算每种分布下 LLM 与 User 的距离指标：
    - 对所有 user-llm 对求最短路距离；
    - 取这些距离的平均值 d_mean；
    - 使用 sigmoid(d_mean) 将其归一到 (0, 1)。
    """
    import networkx as nx
    import math

    G = nx.DiGraph()
    for nid in network.nodes:
        G.add_node(nid)

    for src, links in network.links.items():
        for link in links:
            if link.is_reverse:
                continue
            G.add_edge(src, link.dst, weight=link.distance)

    distances = []
    for uid in users.keys():
        for lid in llms.keys():
            try:
                d = nx.shortest_path_length(G,
                                            source=uid,
                                            target=lid,
                                            weight='weight')
                distances.append(float(d))
            except nx.NetworkXNoPath:
                continue

    if not distances:
        return 0.0

    d_mean = float(np.mean(distances))
    return 1.0 / (1.0 + math.exp(-d_mean))


def set_uniform_bandwidth(network, bandwidth_value):
    """将网络中所有链路的带宽设置为统一值"""
    for src, links in network.links.items():
        for link in links:
            if not link.is_reverse:
                link.capacity = bandwidth_value


def set_uniform_computation(llms, computation_value):
    """将所有LLM的计算资源设置为统一值"""
    for llm in llms.values():
        llm.computation = computation_value
        llm.available_computation = computation_value


def run_algorithm(network, users, llms, algorithm_name, k=None):
    """
    运行指定算法并返回结果

    返回: (allocations, network, total_cost, service_rate)
    """
    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)
    total_bw = sum(u.bw for u in users.values())

    if algorithm_name == 'no-split':
        allocations = []
        # 按带宽降序依次为用户分配，保证 no-split 是“大流量优先”
        sorted_users = sorted(users.items(),
                              key=lambda item: item[1].bw,
                              reverse=True)
        for uid, u in sorted_users:
            distances, prev = net.dijkstra_with_capacity(uid, u.bw)
            min_cost = float('inf')
            best_llm = None

            for lid, llm in llms_copy.items():
                if distances[
                        lid] < min_cost and llm.available_computation >= u.computation:
                    min_cost = distances[lid]
                    best_llm = lid

            if best_llm is None:
                continue

            node_path, link_path = net.get_path_with_links(prev, uid, best_llm)
            if not node_path:
                continue

            net.send_flow(link_path, u.bw)
            if is_shared:
                llms_copy[best_llm].available_computation -= u.computation

            path_cost = min_cost * u.bw
            allocations.append({
                'algorithm': 'no-split',
                'user_id': uid,
                'llm_id': best_llm,
                'path': node_path,
                'cost': path_cost,
                'flow': u.bw
            })

    elif 'split' in algorithm_name and 'augment' not in algorithm_name:
        # k-split
        u0 = next(iter(users.values()))
        single_bw = u0.bw
        single_cpu = u0.computation

        S, T = -1, -2
        net.add_node(Entity.Node(S, 0, 0))
        net.add_node(Entity.Node(T, 0, 0))

        for uid, u in users.items():
            net.add_link(S, uid, u.bw, 0)

        if is_shared:
            for lid, llm in llms_copy.items():
                max_customers = int(llm.computation // single_cpu)
                net.add_link(lid, T, max_customers * single_bw, 0)
        else:
            for lid, llm in llms_copy.items():
                net.add_link(lid, T, total_bw, 0)

        allocations = []
        remaining = total_bw

        while remaining > 1e-9:
            push = min(k, remaining)
            dist, prev = net.dijkstra_with_capacity(S, push)

            if dist[T] == float('inf'):
                break

            node_path, link_path = net.get_path_with_links(prev, S, T)
            if not node_path:
                break

            user_id = node_path[1]
            llm_id = node_path[-2]

            net.send_flow(link_path, push)
            if is_shared:
                llms_copy[llm_id].available_computation -= single_cpu

            path_cost = dist[T] * push
            allocations.append({
                'algorithm': f'{k}-split',
                'user_id': user_id,
                'llm_id': llm_id,
                'path': node_path[1:-1],
                'cost': path_cost,
                'flow': push
            })
            remaining -= push

    elif 'augment' in algorithm_name and 'bottleneck' not in algorithm_name:
        # k-split-augment
        u0 = next(iter(users.values()))
        single_bw = u0.bw
        single_cpu = u0.computation

        S, T = -1, -2
        net.add_node(Entity.Node(S, 0, 0))
        net.add_node(Entity.Node(T, 0, 0))

        for uid, u in users.items():
            net.add_link(S, uid, u.bw, 0)

        if is_shared:
            for lid, llm in llms_copy.items():
                max_customers = int(llm.computation // single_cpu)
                net.add_link(lid, T, max_customers * single_bw, 0)
        else:
            for lid, llm in llms_copy.items():
                net.add_link(lid, T, total_bw, 0)

        # 先运行最小费用流更新边上最终流量，再基于最终流动重新分解真实路径
        _, _, _, _ = net.successive_shortest_paths(S, T, total_bw, k=k)
        allocations = net.decompose_flow_paths(S, T, f'{k}-split-augment')

        if is_shared and single_bw > 0:
            for allocation in allocations:
                llm_id = allocation['llm_id']
                flow_units = allocation['flow'] / single_bw
                llms_copy[
                    llm_id].available_computation -= single_cpu * flow_units

    elif algorithm_name == 'bottleneck-augment':
        u0 = next(iter(users.values()))
        single_bw = u0.bw
        single_cpu = u0.computation

        S, T = -1, -2
        net.add_node(Entity.Node(S, 0, 0))
        net.add_node(Entity.Node(T, 0, 0))

        for uid, u in users.items():
            net.add_link(S, uid, u.bw, 0)

        if is_shared:
            for lid, llm in llms_copy.items():
                max_customers = int(llm.computation // single_cpu)
                net.add_link(lid, T, max_customers * single_bw, 0)
        else:
            for lid, llm in llms_copy.items():
                net.add_link(lid, T, total_bw, 0)

        # 使用瓶颈流量推进最小费用流，结束后按最终流重新分解路径
        _, _, _, _ = net.successive_shortest_paths(S,
                                                   T,
                                                   total_bw,
                                                   k=1,
                                                   use_bottleneck=True)
        allocations = net.decompose_flow_paths(S, T, 'bottleneck-augment')

        if is_shared and single_bw > 0:
            for allocation in allocations:
                llm_id = allocation['llm_id']
                flow_units = allocation['flow'] / single_bw
                llms_copy[
                    llm_id].available_computation -= single_cpu * flow_units

    total_cost = sum(a['cost'] for a in allocations)
    total_allocated = sum(a['flow'] for a in allocations)
    service_rate = total_allocated / total_bw if total_bw > 0 else 0

    return allocations, net, total_cost, service_rate


def compute_user_distance_flow_ratio(allocations, users, user_order=None):
    """
    计算每个用户节点所有路径总距离与其被服务流量总量的比值。

    返回:
        user_ids: List[int]
        ratios: List[float]
        served_flows: List[float]
    """
    user_total_cost = {uid: 0.0 for uid in users.keys()}
    user_total_flow = {uid: 0.0 for uid in users.keys()}

    for allocation in allocations:
        uid = allocation['user_id']
        if uid not in user_total_cost:
            continue
        flow = allocation['flow']
        cost = allocation['cost']
        user_total_cost[uid] += cost
        user_total_flow[uid] += flow

    if user_order is not None:
        ordered_user_ids = list(user_order)
        existing = set(ordered_user_ids)
        for uid in users.keys():
            if uid not in existing:
                ordered_user_ids.append(uid)
    else:
        ordered_user_ids = []
        seen = set()
        for allocation in allocations:
            uid = allocation['user_id']
            if uid in users and uid not in seen:
                seen.add(uid)
                ordered_user_ids.append(uid)
        for uid in users.keys():
            if uid not in seen:
                ordered_user_ids.append(uid)

    ratios = []
    served_flows = []
    for uid in ordered_user_ids:
        total_flow = user_total_flow.get(uid, 0.0)
        total_cost = user_total_cost.get(uid, 0.0)
        if total_flow > 0:
            ratio = total_cost / total_flow
        else:
            ratio = 0.0
        ratios.append(ratio)
        served_flows.append(total_flow)

    return ordered_user_ids, ratios, served_flows


def run_augment_with_compute_flow_mapping(network,
                                          users,
                                          llms,
                                          algorithm_name,
                                          k=None,
                                          flow_per_compute: float = 125.0):
    """
    在不改动底层 Entity.Network 实现的前提下，基于
    “1 计算需求 = flow_per_compute 单位流量” 的映射，
    为 augment 系列算法（k-split-augment / bottleneck-augment）
    构造超源/超汇网络并调用 SSP + 路径分解。

    返回 (allocations, net, total_cost, service_rate)
    """
    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    S, T = -1, -2
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 用户侧：按计算需求映射成总流量需求
    epsilon_cost = 1e-6
    total_flow_demand = 0.0
    for uid, u in users.items():
        demand_flow = u.bw
        if demand_flow <= 0:
            continue
        total_flow_demand += demand_flow
        net.add_link(S, uid, demand_flow, epsilon_cost)

    if total_flow_demand <= 0:
        return [], net, 0.0, 0.0

    # LLM 侧：每 1 计算单位可以支撑 flow_per_compute 单位流量
    if is_shared:
        for lid, llm in llms_copy.items():
            cap_flow = float(llm.computation) * flow_per_compute
            if cap_flow <= 0:
                continue
            net.add_link(lid, T, cap_flow, epsilon_cost)
    else:
        for lid, llm in llms_copy.items():
            net.add_link(lid, T, total_flow_demand, epsilon_cost)

    use_bottleneck = (algorithm_name == 'bottleneck-augment')

    if use_bottleneck:
        _, _, _, _ = net.successive_shortest_paths(S,
                                                   T,
                                                   total_flow_demand,
                                                   k=1,
                                                   use_bottleneck=True)
        decompose_name = 'bottleneck-augment'
    else:
        if k is None:
            raise ValueError("k must be provided for k-split-augment")
        _, _, _, _ = net.successive_shortest_paths(S,
                                                   T,
                                                   total_flow_demand,
                                                   k=k)
        decompose_name = f'{k}-split-augment'

    allocations = net.decompose_flow_paths(S, T, decompose_name)

    # 按映射扣减 LLM 计算资源：每 flow_per_compute 流量对应 1 计算单位
    if is_shared and flow_per_compute > 0:
        for allocation in allocations:
            llm_id = allocation['llm_id']
            compute_used = allocation['flow'] / flow_per_compute
            if llm_id in llms_copy:
                llms_copy[llm_id].available_computation -= compute_used

    total_cost = sum(a['cost'] for a in allocations)
    total_allocated = sum(a['flow'] for a in allocations)
    service_rate = (total_allocated /
                    total_flow_demand) if total_flow_demand > 0 else 0.0

    return allocations, net, total_cost, service_rate


def run_k_split_with_compute_flow_mapping(network,
                                          users,
                                          llms,
                                          k: int,
                                          flow_per_compute: float = 100.0):
    """
    基于“1 计算 = flow_per_compute 流量”的映射，实现 k-split 版本：
    - S->user 容量 = user.computation * flow_per_compute
    - llm->T 容量 = llm.computation * flow_per_compute
    - 每次增广推送 k 个“计算单位”，即 k * flow_per_compute 的流量
    """
    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    S, T = -1, -2
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    total_flow_demand = 0.0
    for uid, u in users.items():
        demand_flow = u.bw
        if demand_flow <= 0:
            continue
        total_flow_demand += demand_flow
        net.add_link(S, uid, demand_flow, 0)

    if total_flow_demand <= 0:
        return [], net, 0.0, 0.0

    if is_shared:
        for lid, llm in llms_copy.items():
            cap_flow = float(llm.computation) * flow_per_compute
            if cap_flow <= 0:
                continue
            net.add_link(lid, T, cap_flow, 0)
    else:
        for lid, llm in llms_copy.items():
            net.add_link(lid, T, total_flow_demand, 0)

    allocations = []
    remaining = total_flow_demand

    while remaining > 1e-9:
        push = min(k * flow_per_compute, remaining)
        dist, prev = net.dijkstra_with_capacity(S, push)

        if dist[T] == float('inf'):
            break

        node_path, link_path = net.get_path_with_links(prev, S, T)
        if not node_path:
            break

        user_id = node_path[1]
        llm_id = node_path[-2]

        net.send_flow(link_path, push)
        if is_shared and flow_per_compute > 0:
            compute_used = push / flow_per_compute
            llms_copy[llm_id].available_computation -= compute_used

        path_cost = dist[T] * push
        allocations.append({
            'algorithm': f'{k}-split',
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': path_cost,
            'flow': push
        })
        remaining -= push

    total_cost = sum(a['cost'] for a in allocations)
    total_allocated = sum(a['flow'] for a in allocations)
    service_rate = (total_allocated /
                    total_flow_demand) if total_flow_demand > 0 else 0.0

    return allocations, net, total_cost, service_rate


def run_k_split_with_compute_flow_mapping_v2(network, users, llms, k: int):
    """
    k-split 在第三循环下的“计算 → 流量”映射版本：
    - S->user 容量 = user.computation，总计算需求，距离为极小值；
    - llm->T 容量 = llm.computation * 125，距离为极小值；
    - 每次增广推送 k 个“计算单位”（即 k 的流量）。
    """
    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    S, T = -1, -2
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    epsilon_cost = 1e-6
    total_flow_demand = 0.0
    for uid, u in users.items():
        demand = float(u.computation)
        if demand <= 0:
            continue
        total_flow_demand += demand
        net.add_link(S, uid, demand, epsilon_cost)

    if total_flow_demand <= 0:
        return [], net, 0.0, 0.0

    flow_per_compute = 125.0
    if is_shared:
        for lid, llm in llms_copy.items():
            cap = float(llm.computation) * flow_per_compute
            if cap <= 0:
                continue
            net.add_link(lid, T, cap, epsilon_cost)
    else:
        for lid, llm in llms_copy.items():
            net.add_link(lid, T, total_flow_demand, epsilon_cost)

    allocations = []
    remaining = total_flow_demand

    while remaining > 1e-9:
        push = min(float(k), remaining)
        dist, prev = net.dijkstra_with_capacity(S, push)

        if dist[T] == float('inf'):
            break

        node_path, link_path = net.get_path_with_links(prev, S, T)
        if not node_path:
            break

        user_id = node_path[1]
        llm_id = node_path[-2]

        net.send_flow(link_path, push)

        path_cost = dist[T] * push
        allocations.append({
            'algorithm': f'{k}-split',
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': path_cost,
            'flow': push
        })
        remaining -= push

    total_cost = sum(a['cost'] for a in allocations)
    total_allocated = sum(a['flow'] for a in allocations)
    service_rate = (total_allocated /
                    total_flow_demand) if total_flow_demand > 0 else 0.0

    return allocations, net, total_cost, service_rate


def plot_user_distance_flow_ratio(user_labels, ratios_by_algorithm,
                                  service_rates_by_algorithm, title, filename):
    """
    绘制用户维度的路径距离与服务流量比值图，并在右Y轴上显示服务率。
    """
    x_positions = np.arange(len(user_labels))

    # 兼容单算法调用场景：如果传入的是列表和标量，则封装为单算法字典
    if not isinstance(ratios_by_algorithm, dict):
        ratios_by_algorithm = {'100-split-augment': ratios_by_algorithm}
    if not isinstance(service_rates_by_algorithm, dict):
        service_rates_by_algorithm = {
            '100-split-augment': service_rates_by_algorithm
        }

    fig, ax1 = plt.subplots(figsize=(16, 8))

    algorithms = [
        'no-split', '1-split', '100-split-augment', 'bottleneck-augment'
    ]
    colors = {
        'no-split': 'tab:blue',
        '1-split': 'tab:red',
        '100-split-augment': 'tab:purple',
        'bottleneck-augment': 'tab:pink'
    }
    linestyles = {
        'no-split': '-',
        '1-split': '--',
        '100-split-augment': '-.',
        'bottleneck-augment': ':'
    }
    markers = {
        'no-split': 'o',
        '1-split': 's',
        '100-split-augment': '^',
        'bottleneck-augment': 'D'
    }

    for alg in algorithms:
        if alg not in ratios_by_algorithm:
            continue
        ax1.plot(x_positions,
                 ratios_by_algorithm[alg],
                 marker=markers.get(alg, 'o'),
                 color=colors.get(alg, 'gray'),
                 linewidth=2.0,
                 label=f'{alg} (ratio)',
                 linestyle=linestyles.get(alg, '-'))
    ax1.set_xlabel('User (bandwidth)', fontsize=13)
    ax1.set_ylabel('Avg path distance per unit flow', fontsize=13)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(user_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    for alg in algorithms:
        if alg not in service_rates_by_algorithm:
            continue
        service_rate_percent = service_rates_by_algorithm[alg] * 100
        ax2.plot(x_positions, [service_rate_percent] * len(x_positions),
                 linestyle=linestyles.get(alg, '--'),
                 color=colors.get(alg, 'gray'),
                 linewidth=1.5,
                 alpha=0.8,
                 label=f'{alg} (sr)')
    ax2.set_ylabel('Service Rate (%)', fontsize=13)
    ax2.set_ylim([0, 105])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = []
    all_labels = []
    for alg in algorithms:
        ratio_label = f'{alg} (ratio)'
        if ratio_label in labels1:
            idx = labels1.index(ratio_label)
            all_handles.append(handles1[idx])
            all_labels.append(labels1[idx])

        sr_label = f'{alg} (sr)'
        if sr_label in labels2:
            idx = labels2.index(sr_label)
            all_handles.append(handles2[idx])
            all_labels.append(labels2[idx])

    if all_handles:
        ax1.legend(all_handles,
                   all_labels,
                   fontsize=9,
                   loc='upper left',
                   bbox_to_anchor=(1.15, 1))

    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_user_distance_flow_ratio_bars(user_labels, ratios_by_algorithm,
                                       service_rates_by_algorithm, title,
                                       filename):
    """
    绘制用户维度的“总距离 / 被服务流量”柱状图：
    - X 轴为用户（按照 no-split 下的分配顺序排列）
    - 每个 X 位置有 4 根柱子：no-split, 1-split, 1-split-augment, bottleneck-augment
    - 右 Y 轴叠加各算法的服务率折线（仅作趋势参考）
    """
    x_positions = np.arange(len(user_labels))

    fig, ax1 = plt.subplots(figsize=(16, 8))

    algorithms = [
        'no-split', '1-split', '1-split-augment', 'bottleneck-augment'
    ]
    colors = {
        'no-split': 'tab:blue',
        '1-split': 'tab:red',
        '1-split-augment': 'tab:green',
        'bottleneck-augment': 'tab:pink',
    }
    linestyles = {
        'no-split': '-',
        '1-split': '--',
        '1-split-augment': '-.',
        'bottleneck-augment': ':',
    }
    markers = {
        'no-split': 'o',
        '1-split': 's',
        '1-split-augment': '^',
        'bottleneck-augment': 'D',
    }

    width = 0.8 / len(algorithms)

    for idx, alg in enumerate(algorithms):
        if alg not in ratios_by_algorithm:
            continue
        ratios = ratios_by_algorithm[alg]
        offset = (idx - len(algorithms) / 2) * width + width / 2
        ax1.bar(x_positions + offset,
                ratios,
                width,
                label=f'{alg} (ratio)',
                color=colors.get(alg, 'gray'),
                alpha=0.7)

    ax1.set_xlabel('User (bandwidth)', fontsize=13)
    ax1.set_ylabel('Avg path distance per unit flow', fontsize=13)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(user_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    for alg in algorithms:
        if alg not in service_rates_by_algorithm:
            continue
        service_rate_percent = service_rates_by_algorithm[alg] * 100
        ax2.plot(
            x_positions,
            [service_rate_percent] * len(x_positions),
            linestyle=linestyles.get(alg, '--'),
            color=colors.get(alg, 'gray'),
            linewidth=1.5,
            alpha=0.9,
            label=f'{alg} (service rate)',
        )
    ax2.set_ylabel('Service Rate (%)', fontsize=13)
    ax2.set_ylim([0, 105])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = []
    all_labels = []
    for label, handle in zip(labels1, handles1):
        all_labels.append(label)
        all_handles.append(handle)
    for label, handle in zip(labels2, handles2):
        all_labels.append(label)
        all_handles.append(handle)

    ax1.legend(all_handles,
               all_labels,
               fontsize=9,
               loc='upper left',
               bbox_to_anchor=(1.15, 1))

    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dual_axis(x_values, data_dict, x_label, title, filename):
    """
    绘制双Y轴图表

    参数:
        x_values: X轴值列表
        data_dict: {algorithm: [(opt1, sr1), (opt2, sr2), ...]}
        x_label: X轴标签
        title: 图表标题
        filename: 保存文件名
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))

    algorithms = [
        'no-split', '1-split', '100-split-augment', 'bottleneck-augment'
    ]
    colors = {
        'no-split': 'tab:blue',
        '1-split': 'tab:red',
        '100-split-augment': 'tab:purple',
        'bottleneck-augment': 'tab:pink'
    }

    # 不同线型以区分重叠的折线
    linestyles = {
        'no-split': '-',
        '1-split': '--',
        '100-split-augment': '-.',
        'bottleneck-augment': ':'
    }

    markers = {
        'no-split': 'o',
        '1-split': 's',
        '100-split-augment': '^',
        'bottleneck-augment': 'D'
    }

    # 左Y轴：cost（真实总开销）
    x_positions = np.arange(len(x_values))
    width = 0.8 / len(algorithms)

    for idx, alg in enumerate(algorithms):
        if alg not in data_dict:
            continue

        # data_dict[alg][i] = (optimization, service_rate, total_cost)
        optimizations = [data_dict[alg][i][0] for i in range(len(x_values))]
        costs = [data_dict[alg][i][2] for i in range(len(x_values))]
        offset = (idx - len(algorithms) / 2) * width + width / 2

        ax1.bar(x_positions + offset,
                costs,
                width,
                label=f'{alg} (cost)',
                color=colors.get(alg, 'gray'),
                alpha=0.7)

    ax1.set_xlabel(x_label, fontsize=13)
    ax1.set_ylabel('Total Cost', fontsize=13)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_values, rotation=45, ha='right')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右Y轴：optimization（服务率只以折线形式叠加，不单独给Y轴）
    ax2 = ax1.twinx()

    for idx, alg in enumerate(algorithms):
        if alg not in data_dict:
            continue

        optimizations = [data_dict[alg][i][0] for i in range(len(x_values))]
        ax2.plot(x_positions,
                 optimizations,
                 marker=markers.get(alg, 'o'),
                 color=colors.get(alg, 'gray'),
                 label=f'{alg} (opt)',
                 linewidth=2.5,
                 markersize=6,
                 alpha=0.9,
                 linestyle=linestyles.get(alg, '-'))

    # 在右Y轴上叠加服务率折线，不增加新的Y轴标签
    for idx, alg in enumerate(algorithms):
        if alg not in data_dict:
            continue

        service_rates = [
            data_dict[alg][i][1] * 100 for i in range(len(x_values))
        ]
        ax2.plot(x_positions,
                 service_rates,
                 marker=None,
                 color=colors.get(alg, 'gray'),
                 label=f'{alg} (sr)',
                 linewidth=1.5,
                 alpha=0.9,
                 linestyle=linestyles.get(alg, '-'))

    ax2.set_ylabel('Optimization vs 1-split-augment (%)', fontsize=13)

    # 合并图例，确保顺序正确
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # 按算法顺序重新排列图例
    all_handles = []
    all_labels = []
    for alg in algorithms:
        # 添加cost柱状图
        cost_label = f'{alg} (cost)'
        if cost_label in labels1:
            idx = labels1.index(cost_label)
            all_handles.append(handles1[idx])
            all_labels.append(labels1[idx])

        # 添加optimization折线图
        opt_label = f'{alg} (opt)'
        if opt_label in labels2:
            idx = labels2.index(opt_label)
            all_handles.append(handles2[idx])
            all_labels.append(labels2[idx])

        # 添加service rate折线图（无独立Y轴，仅用于趋势）
        sr_label = f'{alg} (sr)'
        if sr_label in labels2:
            idx = labels2.index(sr_label)
            all_handles.append(handles2[idx])
            all_labels.append(labels2[idx])

    ax1.legend(all_handles,
               all_labels,
               fontsize=9,
               loc='upper left',
               bbox_to_anchor=(1.15, 1))

    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    已保存: {filename}")


def save_to_excel_multi_sheet(all_distributions_data, filename):
    """
    保存所有分布的数据到一个Excel文件的不同sheet

    参数:
        all_distributions_data: {distribution_name: [rows]}
        filename: Excel文件路径
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, rows in all_distributions_data.items():
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  已保存所有分布到Excel: {filename}")


def run_user_pattern_sweep():
    """
    用户计算需求分布 × 带宽 × LLM容量的三重扫描：
    1. 对每个 pattern，收集所有 (distribution, bandwidth, computation) 组合的数据
    2. 生成带宽变化图（保存到 pattern{i}/bandwidth/{distribution}/）
    3. 生成LLM容量变化图（保存到 pattern{i}/llm/{distribution}/）
    4. 生成用户级柱状图（保存到 pattern{i}/user_level/{distribution}/）
    5. 保存数据到 userpattern/bandwidth.xlsx, llm.xlsx, user.xlsx
    """
    # 三类视角的数据收集字典：{sheet_name(distribution): [rows]}
    bandwidth_all_data = {}
    llm_all_data = {}
    user_all_data = {}

    print("运行用户计算需求分布扫描...")

    for pattern_index, pattern in enumerate(USER_COMPUTATION_PATTERNS):
        print(f"\n处理 Pattern {pattern_index + 1}: {pattern}")

        # 创建pattern目录
        pattern_dir = os.path.join(_USER_PATTERN_DIR,
                                   f'pattern{pattern_index + 1}')
        bandwidth_base_dir = os.path.join(pattern_dir, 'bandwidth')
        llm_base_dir = os.path.join(pattern_dir, 'llm')
        user_level_base_dir = os.path.join(pattern_dir, 'user_level')

        os.makedirs(bandwidth_base_dir, exist_ok=True)
        os.makedirs(llm_base_dir, exist_ok=True)
        os.makedirs(user_level_base_dir, exist_ok=True)

        for user_distribution in DISTRIBUTION_TYPES:
            for llm_distribution in DISTRIBUTION_TYPES:
                distribution_name = f'{user_distribution}-{llm_distribution}'
                print(f"  处理分布: {distribution_name}")

                # 在各个基础目录下创建分布子目录
                bandwidth_dir = os.path.join(bandwidth_base_dir,
                                             distribution_name)
                llm_dir = os.path.join(llm_base_dir, distribution_name)
                user_level_dir = os.path.join(user_level_base_dir,
                                              distribution_name)

                os.makedirs(bandwidth_dir, exist_ok=True)
                os.makedirs(llm_dir, exist_ok=True)
                os.makedirs(user_level_dir, exist_ok=True)

                # 收集所有 (bandwidth, computation) 组合的结果
                all_results = {
                }  # key: (bandwidth, computation), value: {alg: (opt, sr, cost)}
                all_user_data = {
                }  # key: (bandwidth, computation), value: (user_data_results, user_ids_ordered, users)

                # 运行所有组合
                for bandwidth in BANDWIDTH_VALUES:
                    for computation in COMPUTATION_VALUES:
                        # 加载网络和数据
                        json = Entity.load_network_from_sheets()
                        network = json['network']
                        llms = Entity.load_llm_info(user_distribution,
                                                    llm_distribution)
                        users = Entity.load_user_info(user_distribution)

                        users = dict(
                            sorted(users.items(),
                                   key=lambda item: item[1].bw,
                                   reverse=True))

                        # 应用用户计算需求分布模式
                        user_ids = list(users.keys())
                        for idx, demand in enumerate(pattern):
                            user_id = user_ids[idx]
                            user = users[user_id]
                            user.computation = float(demand)
                            bw_value = USER_COMPUTATION_TO_BANDWIDTH.get(
                                demand)
                            user.bw = float(bw_value)

                        # 设置统一参数
                        set_uniform_bandwidth(network, bandwidth)
                        set_uniform_computation(llms, computation)

                        # 用户节点按带宽降序排列
                        user_ids_ordered = sorted(
                            users.keys(),
                            key=lambda uid: users[uid].bw,
                            reverse=True)

                        # 运行四种算法
                        algorithm_results = {}
                        user_data_results = {}

                        # 计算 baseline（1-split-augment）
                        _, _, baseline_cost, baseline_sr = run_augment_with_compute_flow_mapping(
                            network,
                            users,
                            llms,
                            '1-split-augment',
                            k=1,
                            flow_per_compute=125.0)

                        user_algorithms = [
                            'no-split', '1-split', '1-split-augment',
                            'bottleneck-augment'
                        ]

                        for alg in user_algorithms:
                            if alg == 'no-split':
                                allocations, _, cost, sr = run_algorithm(
                                    network, users, llms, 'no-split')
                            elif alg == '1-split':
                                allocations, _, cost, sr = run_k_split_with_compute_flow_mapping(
                                    network,
                                    users,
                                    llms,
                                    k=1,
                                    flow_per_compute=125.0)
                            elif alg == '1-split-augment':
                                allocations, _, cost, sr = run_augment_with_compute_flow_mapping(
                                    network,
                                    users,
                                    llms,
                                    '1-split-augment',
                                    k=1,
                                    flow_per_compute=125.0)
                            elif alg == 'bottleneck-augment':
                                allocations, _, cost, sr = run_augment_with_compute_flow_mapping(
                                    network,
                                    users,
                                    llms,
                                    'bottleneck-augment',
                                    k=1,
                                    flow_per_compute=125.0)
                            else:
                                continue

                            # 计算 optimization
                            if baseline_cost > 0 and cost > 0:
                                opt_percentage = (cost -
                                                  baseline_cost) / cost * 100
                            else:
                                opt_percentage = 0

                            algorithm_results[alg] = (opt_percentage, sr, cost)

                            # 计算用户级指标
                            _, ratios, served_flows = compute_user_distance_flow_ratio(
                                allocations,
                                users,
                                user_order=user_ids_ordered)
                            user_data_results[alg] = (ratios, served_flows, sr)

                        # 保存结果
                        all_results[(bandwidth,
                                     computation)] = algorithm_results
                        all_user_data[(bandwidth,
                                       computation)] = (user_data_results,
                                                        user_ids_ordered,
                                                        users)

                # 生成带宽视角数据（循环1逻辑）
                for bandwidth in BANDWIDTH_VALUES:
                    data_by_algorithm = {}

                    for computation in COMPUTATION_VALUES:
                        results = all_results.get((bandwidth, computation))
                        if not results:
                            continue

                        for alg in [
                                'no-split', '1-split', '100-split-augment',
                                'bottleneck-augment'
                        ]:
                            if alg == '100-split-augment':
                                alg_key = '1-split-augment'
                            else:
                                alg_key = alg

                            if alg_key not in results:
                                continue

                            if alg not in data_by_algorithm:
                                data_by_algorithm[alg] = []

                            data_by_algorithm[alg].append(results[alg_key])

                    if data_by_algorithm:
                        # 仅记录到 bandwidth 视角数据，具体绘图由重绘脚本完成
                        rows = bandwidth_all_data.setdefault(
                            distribution_name, [])
                        for alg in [
                                'no-split', '1-split', '100-split-augment',
                                'bottleneck-augment'
                        ]:
                            if alg not in data_by_algorithm:
                                continue
                            for idx, computation in enumerate(
                                    COMPUTATION_VALUES):
                                if idx < len(data_by_algorithm[alg]):
                                    opt, sr, cost = data_by_algorithm[alg][idx]
                                    rows.append({
                                        'pattern_index':
                                        pattern_index + 1,
                                        'pattern':
                                        ','.join(str(v) for v in pattern),
                                        'bandwidth':
                                        bandwidth,
                                        'llm_computation':
                                        computation,
                                        'algorithm':
                                        alg,
                                        'optimization':
                                        opt,
                                        'service_rate':
                                        sr,
                                        'total_cost':
                                        cost
                                    })

                # 生成 LLM 容量视角数据（循环2逻辑）
                for computation in COMPUTATION_VALUES:
                    data_by_algorithm = {}

                    for bandwidth in BANDWIDTH_VALUES:
                        results = all_results.get((bandwidth, computation))
                        if not results:
                            continue

                        for alg in [
                                'no-split', '1-split', '100-split-augment',
                                'bottleneck-augment'
                        ]:
                            if alg == '100-split-augment':
                                alg_key = '1-split-augment'
                            else:
                                alg_key = alg

                            if alg_key not in results:
                                continue

                            if alg not in data_by_algorithm:
                                data_by_algorithm[alg] = []

                            data_by_algorithm[alg].append(results[alg_key])

                    if data_by_algorithm:
                        # 仅记录到 llm 视角数据，具体绘图由重绘脚本完成
                        rows = llm_all_data.setdefault(distribution_name, [])
                        for alg in [
                                'no-split', '1-split', '100-split-augment',
                                'bottleneck-augment'
                        ]:
                            if alg not in data_by_algorithm:
                                continue
                            for idx, bandwidth in enumerate(BANDWIDTH_VALUES):
                                if idx < len(data_by_algorithm[alg]):
                                    opt, sr, cost = data_by_algorithm[alg][idx]
                                    rows.append({
                                        'pattern_index':
                                        pattern_index + 1,
                                        'pattern':
                                        ','.join(str(v) for v in pattern),
                                        'llm_computation':
                                        computation,
                                        'bandwidth':
                                        bandwidth,
                                        'algorithm':
                                        alg,
                                        'optimization':
                                        opt,
                                        'service_rate':
                                        sr,
                                        'total_cost':
                                        cost
                                    })

                # 生成用户级数据（不在此处绘图）
                for bandwidth in BANDWIDTH_VALUES:
                    for computation in COMPUTATION_VALUES:
                        user_data_tuple = all_user_data.get(
                            (bandwidth, computation))
                        if not user_data_tuple:
                            continue

                        user_data_results, user_ids_ordered, users = user_data_tuple

                        # 构造用户标签：序号(带宽)
                        user_labels = []
                        for index, uid in enumerate(user_ids_ordered):
                            bw_value = int(users[uid].bw)
                            user_labels.append(f'{index + 1}({bw_value})')

                        # 提取ratios和service_rates
                        ratios_by_algorithm = {}
                        service_rates_by_algorithm = {}
                        served_flows_by_algorithm = {}

                        for alg in [
                                'no-split', '1-split', '1-split-augment',
                                'bottleneck-augment'
                        ]:
                            if alg not in user_data_results:
                                continue
                            ratios, served_flows, sr = user_data_results[alg]
                            ratios_by_algorithm[alg] = ratios
                            service_rates_by_algorithm[alg] = sr
                            served_flows_by_algorithm[alg] = served_flows

                        # 记录到 user 视角数据，实际绘图由重绘脚本完成
                        rows = user_all_data.setdefault(distribution_name, [])
                        for alg in [
                                'no-split', '1-split', '1-split-augment',
                                'bottleneck-augment'
                        ]:
                            if alg not in ratios_by_algorithm:
                                continue

                            ratios = ratios_by_algorithm[alg]
                            served_flows = served_flows_by_algorithm[alg]
                            service_rate = service_rates_by_algorithm[alg]

                            for index, uid in enumerate(user_ids_ordered):
                                distance_per_unit_flow = ratios[
                                    index] if index < len(ratios) else 0.0
                                served_flow = served_flows[
                                    index] if index < len(
                                        served_flows) else 0.0
                                rows.append({
                                    'pattern_index':
                                    pattern_index + 1,
                                    'pattern':
                                    ','.join(str(v) for v in pattern),
                                    'bandwidth':
                                    bandwidth,
                                    'llm_computation':
                                    computation,
                                    'user_index':
                                    index + 1,
                                    'user_id':
                                    uid,
                                    'user_bandwidth':
                                    users[uid].bw,
                                    'user_computation':
                                    users[uid].computation,
                                    'algorithm':
                                    alg,
                                    'distance_per_unit_flow':
                                    distance_per_unit_flow,
                                    'served_flow':
                                    served_flow,
                                    'service_rate':
                                    service_rate
                                })

    # 将 bandwidth / llm / user 三类视角汇总到一个 Excel：
    # 通过 view 字段区分来源，sheet 仍按 distribution 命名
    combined_data = {}

    for distribution_name, rows in bandwidth_all_data.items():
        target_rows = combined_data.setdefault(distribution_name, [])
        for row in rows:
            new_row = dict(row)
            new_row['view'] = 'bandwidth'
            target_rows.append(new_row)

    for distribution_name, rows in llm_all_data.items():
        target_rows = combined_data.setdefault(distribution_name, [])
        for row in rows:
            new_row = dict(row)
            new_row['view'] = 'llm'
            target_rows.append(new_row)

    for distribution_name, rows in user_all_data.items():
        target_rows = combined_data.setdefault(distribution_name, [])
        for row in rows:
            new_row = dict(row)
            new_row['view'] = 'user'
            target_rows.append(new_row)

    if combined_data:
        combined_excel_file = os.path.join(_USER_PATTERN_DIR,
                                           'userpattern_all.xlsx')
        save_to_excel_multi_sheet(combined_data, combined_excel_file)
        print(f"\n所有扫描完成！已写入: {combined_excel_file}")
    else:
        print("\n所有扫描完成，但没有可写入的数据！")


def main():
    """
    整合的用户分布扫描：
    对每个pattern，生成带宽变化图、LLM容量变化图
    （当前文件只负责产生 userpattern_all.xlsx，所有绘图在独立脚本中完成）
    """
    run_user_pattern_sweep()

    return


if __name__ == "__main__":
    main()
