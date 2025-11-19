import Entity
import pandas as pd
import numpy as np
import math
import time
import networkx as nx
import os
import copy

DISTRIBUTION_TYPES = Entity.DISTRIBUTION_TYPES
script_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(script_path, 'results')
is_shared = 0

_RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(_RESULT_DIR, exist_ok=True)
origin_file = os.path.join(_RESULT_DIR, 'OriginResults.xlsx')
dash_file = os.path.join(_RESULT_DIR, 'DashBoard.xlsx')
link_util_file = os.path.join(_RESULT_DIR, 'LinkUtilization.xlsx')
runtime_file = os.path.join(_RESULT_DIR, 'runtime.xlsx')


def merge_allocations_by_path(allocations):
    if not allocations:
        return pd.DataFrame(columns=[
            'algorithm', 'user_id', 'llm_id', 'path', 'total_flow',
            'total_cost', 'original_order'
        ])

    rows = []
    for idx, a in enumerate(allocations):
        path_list = a.get('path', [])
        path_str = '->'.join(map(str, path_list))
        rows.append({
            'algorithm': a.get('algorithm', ''),
            'user_id': a.get('user_id', None),
            'llm_id': a.get('llm_id', None),
            'path': path_str,
            'flow': a.get('flow', 0.0),
            'cost': a.get('cost', 0.0),
            'original_order': idx  # 记录原始顺序
        })

    df = pd.DataFrame(rows)
    grouped = (df.groupby(['algorithm', 'path', 'user_id', 'llm_id'],
                          dropna=False).agg(
                              total_flow=('flow', 'sum'),
                              total_cost=('cost', 'sum'),
                          ).reset_index())

    return grouped


def write_origin_results(file_path, blocks, user_distribution,
                         llm_distribution):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        sheet_name = f'{user_distribution}-{llm_distribution}'
        start_row = 0

        for title, df in blocks:
            if df is None or df.empty:
                continue

            df_to_write = df.copy()

            # 计算总开销
            total_cost = df_to_write['total_cost'].sum()
            total_flow = df_to_write['total_flow'].sum(
            ) if 'total_flow' in df_to_write.columns else 0.0

            total_row = {
                'algorithm': f"{df_to_write['algorithm'].iloc[0]}-Total",
                'user_id': '',
                'llm_id': '',
                'path': 'Total',
                'total_flow': total_flow,
                'total_cost': total_cost,
            }

            df_with_total = pd.concat(
                [df_to_write, pd.DataFrame([total_row])], ignore_index=True)

            # 写入当前块
            df_with_total.to_excel(writer,
                                   sheet_name=sheet_name,
                                   index=False,
                                   startrow=start_row)

            # 下一个块起始行：当前块行数 + 2（空一行）
            start_row += len(df_with_total) + 2


def compute_edge_betweenness(alo_network, top_n=10):
    """返回按介数中心性排序的前 top_n 条有向链路。"""
    G = nx.DiGraph()
    for link_list in alo_network.links.values():
        for link in link_list:
            if link.is_reverse:
                continue
            if link.src < 0 or link.dst < 0:
                continue
            G.add_edge(link.src,
                       link.dst,
                       capacity=link.capacity,
                       weight=link.distance)

    edge_centrality = nx.edge_betweenness_centrality(G,
                                                     weight='weight',
                                                     normalized=True)
    sorted_edges = sorted(edge_centrality.items(),
                          key=lambda item: item[1],
                          reverse=True)
    return sorted_edges[:top_n]


def calculate_link_utilization(alo_network, target_edges):
    """计算指定链路的利用率 (flow/capacity)。"""
    edge_lookup = {}
    for link_list in alo_network.links.values():
        for link in link_list:
            if link.is_reverse:
                continue
            if link.src < 0 or link.dst < 0:
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


def write_link_utilization_report(file_path, sheet_name, algorithm_utils,
                                  critical_edges):
    columns = ['algorithm']
    for idx in range(1, len(critical_edges) + 1):
        columns.append(f'critical_link_{idx}')
        columns.append(f'critical_link_{idx}_utilization')

    rows = []
    blank_row = {col: '' for col in columns}

    for alg_name, util_map in algorithm_utils:
        row = {col: '' for col in columns}
        row['algorithm'] = alg_name
        for idx, edge in enumerate(critical_edges, start=1):
            label_key = f'critical_link_{idx}'
            value_key = f'critical_link_{idx}_utilization'
            row[label_key] = f'{edge[0]}->{edge[1]}'
            row[value_key] = util_map.get(edge, 0.0)
        rows.append(row)
        rows.append(blank_row.copy())

    if rows:
        rows.pop()  # 移除最后一行空行

    df = pd.DataFrame(rows, columns=columns)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def write_link_utilization_report(file_path, sheet_name, algorithm_utils,
                                  top_edges):
    """
    覆盖前一个实现：
    - 第一列为算法名
    - 后续列为同一组前 top_n 关键链路的利用率
    - 末尾附带每条关键链路的端点与介数中心性
    """
    edge_pairs = [edge for edge, _ in top_edges]
    edge_betweenness = [bet for _, bet in top_edges]

    columns = ['algorithm']
    for idx in range(1, len(edge_pairs) + 1):
        columns.append(f'link_{idx}_utilization')
    for idx in range(1, len(edge_pairs) + 1):
        columns.append(f'link_{idx}_src')
        columns.append(f'link_{idx}_dst')
        columns.append(f'link_{idx}_betweenness')

    rows = []
    for alg_name, util_map in algorithm_utils:
        row = {col: '' for col in columns}
        row['algorithm'] = alg_name
        for idx, ((src, dst), bet) in enumerate(zip(edge_pairs,
                                                    edge_betweenness),
                                                start=1):
            util_key = f'link_{idx}_utilization'
            src_key = f'link_{idx}_src'
            dst_key = f'link_{idx}_dst'
            bet_key = f'link_{idx}_betweenness'
            row[util_key] = util_map.get((src, dst), 0.0)
            row[src_key] = src
            row[dst_key] = dst
            row[bet_key] = bet
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def write_runtime_report(file_path, sheet_name, runtime_data):
    """
    保存算法运行时间报告

    参数:
        file_path: Excel 文件路径
        sheet_name: sheet 名称（格式：user_distribution-llm_distribution）
        runtime_data: 字典，键为算法名，值为运行时间（秒）
    """
    rows = []
    for algorithm, runtime_seconds in runtime_data.items():
        rows.append({
            'algorithm': algorithm,
            'runtime_seconds': runtime_seconds
        })

    df = pd.DataFrame(rows, columns=['algorithm', 'runtime_seconds'])
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def no_split(network, users, llms, user_ideal_llms):
    print("Starting no-split allocation")
    network_copy = copy.deepcopy(network)

    u0 = next(iter(users.values()))
    single_bw = u0.bw
    single_cpu = u0.computation
    total_bw = sum(u.bw for u in users.values())

    S, T = -1, -2
    network_copy.add_node(Entity.Node(S, 0, 0))
    network_copy.add_node(Entity.Node(T, 0, 0))

    for uid, u in users.items():
        network_copy.add_link(S, uid, u.bw, 1e-19)

    if is_shared:
        for lid, llm in llms.items():
            max_customers = int(llm.computation // single_cpu)
            network_copy.add_link(lid, T, max_customers * single_bw, 1e-19)
    else:
        for lid, llm in llms.items():
            network_copy.add_link(lid, T, total_bw, 1e-19)

    allocations = []
    remaining = total_bw
    while remaining >= 1e-9:
        push = 500
        dist, prev, _ = network_copy.dijkstra_with_capacity(S,
                                                            min_capacity=push,
                                                            target_id=T)

        # 如果此时已经无法为剩余流量找到可行路径，则舍弃剩余流量
        if dist[T] == float('inf'):
            break

        node_path, link_path = network_copy.get_path_with_links(prev, S, T)
        if not node_path:
            break
        user_id = node_path[1]
        llm_id = node_path[-2]

        network_copy.send_flow(link_path, push)
        if is_shared:
            llms[llm_id].available_computation -= single_cpu

        allocations.append({
            "algorithm": "no_split",
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': dist[T] * push,
            'flow': push
        })
        remaining -= push
    return allocations, network_copy


def k_split(network, users, llms, k):
    print(f'Starting k-split with k={k}')
    network_copy = copy.deepcopy(network)

    u0 = next(iter(users.values()))
    single_bw = u0.bw
    single_cpu = u0.computation
    total_bw = sum(u.bw for u in users.values())

    S, T = -1, -2
    network_copy.add_node(Entity.Node(S, 0, 0))
    network_copy.add_node(Entity.Node(T, 0, 0))

    for uid, u in users.items():
        network_copy.add_link(S, uid, u.bw, 1e-19)

    if is_shared:
        for lid, llm in llms.items():
            max_customers = int(llm.computation // single_cpu)
            network_copy.add_link(lid, T, max_customers * single_bw, 1e-19)
    else:
        for lid, llm in llms.items():
            network_copy.add_link(lid, T, total_bw, 1e-19)

    allocations = []
    remaining = total_bw
    while remaining >= 1e-9:
        push = min(k, remaining)
        dist, prev, _ = network_copy.dijkstra_with_capacity(S,
                                                            min_capacity=push,
                                                            target_id=T)

        # 如果此时已经无法为剩余流量找到可行路径，则舍弃剩余流量
        if dist[T] == float('inf'):
            break

        node_path, link_path = network_copy.get_path_with_links(prev, S, T)
        if not node_path:
            break
        user_id = node_path[1]
        llm_id = node_path[-2]

        network_copy.send_flow(link_path, push)
        if is_shared:
            llms[llm_id].available_computation -= single_cpu

        allocations.append({
            "algorithm": f"{k}-split",
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': dist[T] * push,
            'flow': push
        })
        remaining -= push

    return allocations, network_copy


def k_split_augment(network, users, llms, k):
    print(f"进入了{k}-split-augment 算法")
    net = copy.deepcopy(network)

    u0 = next(iter(users.values()))
    single_bw = u0.bw
    single_cpu = u0.computation
    total_bw = sum(u.bw for u in users.values())

    S, T = -1, -2
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    for uid, u in users.items():
        net.add_link(S, uid, u.bw, 0)  # 费用 0

    if is_shared:
        for lid, llm in llms.items():
            max_customers = int(llm.computation // single_cpu)
            net.add_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid, llm in llms.items():
            net.add_link(lid, T, total_bw, 0)

    # 先运行最小费用流，得到最终链路流量分布
    net.successive_shortest_paths(S, T, total_bw, k=k)
    allocations = net.decompose_flow_paths(S,
                                           T,
                                           algorithm_name=f"{k}-split-augment")

    # 基于最终流量结果扣减 LLM 计算资源
    if is_shared and single_bw > 0:
        for allocation in allocations:
            llm_id = allocation['llm_id']
            flow_units = allocation['flow'] / single_bw
            llms[llm_id].available_computation -= single_cpu * flow_units

    return allocations, net


if __name__ == "__main__":
    # for user_distribution in DISTRIBUTION_TYPES:
    #     for llm_distribution in DISTRIBUTION_TYPES:
    user_distribution = 'power_law'
    llm_distribution = 'power_law'
    json = Entity.load_network_from_sheets()
    network = json['network']
    nodes_list = list(json['nodes'].values())
    nodes = json['nodes']
    llms = Entity.load_llm_info(user_distribution, llm_distribution)
    users = Entity.load_user_info(user_distribution)
    for llm in llms.values():
        nodes_list[llm.id].role = 'llm'
        nodes_list[llm.id].deployed = 1
    for user in users.values():
        nodes_list[user.id].role = 'user'

    # 用户按带宽排序
    users = dict(
        sorted(users.items(), key=lambda item: item[1].bw, reverse=True))

    user_ideal_llms = {}
    for user in users.values():
        distances, costs = network.dijkstra_ideal(user.id, user.bw)
        sorted_nodes = sorted(distances, key=distances.get)
        ideal_llms = {
            n: costs[n]
            for n in sorted_nodes if nodes_list[n].role == 'llm'
        }
        user_ideal_llms[user.id] = ideal_llms

    # 初始化运行时间记录字典
    runtime_data = {}

    # no_split
    start_time = time.time()
    no_split_allocations, no_split_network = no_split(network, users, llms,
                                                      user_ideal_llms)
    runtime_data['no-split'] = time.time() - start_time
    network.reset_network(llms)

    # k_split
    ks = [1, 2, 4, 5, 10, 20, 25, 50, 100, 250]
    k_split_results = {kid: None for kid in ks}
    k_split_networks = {kid: None for kid in ks}
    for split_k in ks:
        start_time = time.time()
        k_split_allocations, k_split_network = k_split(network,
                                                       users,
                                                       llms,
                                                       k=split_k)
        runtime_data[f'k-split-{split_k}'] = time.time() - start_time
        k_split_results[split_k] = k_split_allocations
        k_split_networks[split_k] = k_split_network
        network.reset_network(llms)

    # k_split_augment
    augment_results = {kid: None for kid in ks}
    augment_networks = {kid: None for kid in ks}
    for split_k in ks:
        start_time = time.time()
        augment_allocations, augment_network = k_split_augment(network,
                                                               users,
                                                               llms,
                                                               k=split_k)
        runtime_data[f'k-split-augment-{split_k}'] = time.time() - start_time
        augment_results[split_k] = augment_allocations
        augment_networks[split_k] = augment_network
        network.reset_network(llms)

    # 组织所有算法的 allocations
    all_blocks = []

    # no_split 结果
    no_split_df = merge_allocations_by_path(no_split_allocations)
    all_blocks.append(('no-split', no_split_df))

    # k_split 结果
    for k_val, allocs in k_split_results.items():
        if not allocs:
            continue
        df_k = merge_allocations_by_path(allocs)
        all_blocks.append((f'k-split-{k_val}', df_k))
    # k_split_augment 结果
    for k_val, allocs in augment_results.items():
        if not allocs:
            continue
        df_k = merge_allocations_by_path(allocs)
        all_blocks.append((f'k-split-augment-{k_val}', df_k))

    write_origin_results(origin_file, all_blocks, user_distribution,
                         llm_distribution)

    # Link utilization report
    top_edges = compute_edge_betweenness(network, top_n=30)
    critical_edges = [edge for edge, _ in top_edges]

    algorithm_networks = [('no-split', no_split_network)]
    algorithm_networks.extend([(f'k-split-{k_val}', net)
                               for k_val, net in k_split_networks.items()
                               if net is not None])
    algorithm_networks.extend([(f'k-split-augment-{k_val}', net)
                               for k_val, net in augment_networks.items()
                               if net is not None])

    algorithm_utils = []
    for alg_name, net in algorithm_networks:
        if not critical_edges:
            break
        util_map = calculate_link_utilization(net, critical_edges)
        algorithm_utils.append((alg_name, util_map))

    if critical_edges and algorithm_utils:
        write_link_utilization_report(
            link_util_file, f'{user_distribution}-{llm_distribution}',
            algorithm_utils, top_edges)

    # 保存运行时间报告
    write_runtime_report(runtime_file,
                         f'{user_distribution}-{llm_distribution}',
                         runtime_data)
    print(f"Runtime report saved to {runtime_file}")
