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
DISTRIBUTION_TYPES = Entity.DISTRIBUTION_TYPES
is_shared = 1
BANDWIDTH_VALUES = list(range(500, 4250, 250))  # 500到4000，步长250
COMPUTATION_VALUES = list(range(8, 34, 2))  # 8到32，步长2
ALGORITHMS = ['no-split', '1-split', '100-split-augment', 'bottleneck-augment']
BAR_COST_LABEL_VERTICAL_OFFSET = 1.0

# 输出目录
_BANDWIDTH_DIR = os.path.join(os.path.dirname(__file__), 'bandwidth')
os.makedirs(_BANDWIDTH_DIR, exist_ok=True)

_LLM_DIR = os.path.join(os.path.dirname(__file__), 'llm')
os.makedirs(_LLM_DIR, exist_ok=True)


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
        for uid, u in users.items():
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

        allocations, _, _, _ = net.successive_shortest_paths(S,
                                                             T,
                                                             total_bw,
                                                             k=k)

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

        allocations, _, _, _ = net.successive_shortest_paths(
            S, T, total_bw, k=1, use_bottleneck=True)

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

    # 左Y轴：optimization
    x_positions = np.arange(len(x_values))
    width = 0.8 / len(algorithms)

    for idx, alg in enumerate(algorithms):
        if alg not in data_dict:
            continue

        optimizations = [data_dict[alg][i][0] for i in range(len(x_values))]
        costs = [data_dict[alg][i][2] for i in range(len(x_values))]
        offset = (idx - len(algorithms) / 2) * width + width / 2

        bars = ax1.bar(x_positions + offset,
                optimizations,
                width,
                label=f'{alg} (opt)',
                color=colors.get(alg, 'gray'),
                alpha=0.7)

        for bar_index, bar in enumerate(bars):
            bar_height = bar.get_height()
            if bar_index < len(costs):
                cost_value = int(round(costs[bar_index]))
            else:
                cost_value = 0

            if bar_height >= 0:
                label_y = bar_height + BAR_COST_LABEL_VERTICAL_OFFSET + idx * BAR_COST_LABEL_VERTICAL_OFFSET
                vertical_alignment = 'bottom'
            else:
                label_y = bar_height - (BAR_COST_LABEL_VERTICAL_OFFSET + idx * BAR_COST_LABEL_VERTICAL_OFFSET)
                vertical_alignment = 'top'

            ax1.text(bar.get_x() + bar.get_width() / 2,
                     label_y,
                     f'{cost_value}',
                     ha='center',
                     va=vertical_alignment,
                     fontsize=8)

    ax1.set_xlabel(x_label, fontsize=13)
    ax1.set_ylabel('Optimization vs 1-split-augment (%)', fontsize=13)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_values, rotation=45, ha='right')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右Y轴：service_rate
    ax2 = ax1.twinx()

    for idx, alg in enumerate(algorithms):
        if alg not in data_dict:
            continue

        service_rates = [
            data_dict[alg][i][1] * 100 for i in range(len(x_values))
        ]
        ax2.plot(x_positions,
                 service_rates,
                 marker=markers.get(alg, 'o'),
                 color=colors.get(alg, 'gray'),
                 label=f'{alg} (sr)',
                 linewidth=2.5,
                 markersize=6,
                 alpha=0.9,
                 linestyle=linestyles.get(alg, '-'))

    ax2.set_ylabel('Service Rate (%)', fontsize=13)
    ax2.set_ylim([0, 105])

    # 合并图例，确保顺序正确
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # 按算法顺序重新排列图例
    all_handles = []
    all_labels = []
    for alg in algorithms:
        # 添加optimization柱状图
        opt_label = f'{alg} (opt)'
        if opt_label in labels1:
            idx = labels1.index(opt_label)
            all_handles.append(handles1[idx])
            all_labels.append(labels1[idx])

        # 添加service rate折线图
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


def main():
    # 循环1: 带宽大循环，嵌套LLM computation
    bandwidth_all_distributions = {}  # {distribution_name: [rows]}

    for user_distribution in DISTRIBUTION_TYPES:
        for llm_distribution in DISTRIBUTION_TYPES:
            distribution_name = f'{user_distribution}-{llm_distribution}'

            # 存储该分布的所有数据
            all_excel_rows = []

            for bandwidth in BANDWIDTH_VALUES:

                # 存储每个computation下所有算法的结果
                data_by_algorithm = {}  # {algorithm: [(opt, sr), ...]}

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

                    # 设置统一参数
                    set_uniform_bandwidth(network, bandwidth)
                    set_uniform_computation(llms, computation)

                    total_bw = sum(u.bw for u in users.values())

                    # 运行所有算法
                    algorithm_results = {}

                    # 1-split-augment作为baseline
                    _, _, baseline_cost, _ = run_algorithm(network, users, llms,
                                                           '1-split-augment', 1)

                    # no-split
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   'no-split')
                    algorithm_results['no-split'] = (cost, sr)

                    # 1-split
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   '1-split', 1)
                    algorithm_results['1-split'] = (cost, sr)

                    # 100-split-augment
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   '100-split-augment', 100)
                    algorithm_results['100-split-augment'] = (cost, sr)

                    # bottleneck-augment
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   'bottleneck-augment')
                    algorithm_results['bottleneck-augment'] = (cost, sr)

                    # 计算optimization（相对于1-split-augment）
                    for alg, (cost, sr) in algorithm_results.items():
                        if alg not in data_by_algorithm:
                            data_by_algorithm[alg] = []

                        if baseline_cost > 0 and cost > 0:
                            opt_percentage = (cost - baseline_cost) / cost * 100
                        else:
                            opt_percentage = 0

                        data_by_algorithm[alg].append((opt_percentage, sr,
                                                       cost))

                        # 记录到Excel行（累积到总列表）
                        all_excel_rows.append({
                            'bandwidth':
                            bandwidth,
                            'llm_computation':
                            computation,
                            'algorithm':
                            alg,
                            'total_cost':
                            cost,
                            'service_rate':
                            sr,
                            'optimization_vs_1split_augment':
                            opt_percentage
                        })

                # 绘制图表
                plot_filename = os.path.join(
                    _BANDWIDTH_DIR, f'{distribution_name}_bw{bandwidth}.png')
                plot_dual_axis(COMPUTATION_VALUES, data_by_algorithm,
                               'LLM Computation',
                               f'{distribution_name} - Bandwidth={bandwidth}',
                               plot_filename)

            # 将该分布的数据添加到总字典
            bandwidth_all_distributions[distribution_name] = all_excel_rows

    # 循环1结束后，保存所有分布到一个Excel文件
    bandwidth_excel_file = os.path.join(_BANDWIDTH_DIR, 'bandwidth.xlsx')
    save_to_excel_multi_sheet(bandwidth_all_distributions,
                               bandwidth_excel_file)

    # 循环2: LLM computation大循环，嵌套带宽
    llm_all_distributions = {}  # {distribution_name: [rows]}

    for user_distribution in DISTRIBUTION_TYPES:
        for llm_distribution in DISTRIBUTION_TYPES:
            distribution_name = f'{user_distribution}-{llm_distribution}'

            # 存储该分布的所有数据
            all_excel_rows = []

            for computation in COMPUTATION_VALUES:

                # 存储每个bandwidth下所有算法的结果
                data_by_algorithm = {}  # {algorithm: [(opt, sr), ...]}

                for bandwidth in BANDWIDTH_VALUES:
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

                    # 设置统一参数
                    set_uniform_bandwidth(network, bandwidth)
                    set_uniform_computation(llms, computation)

                    total_bw = sum(u.bw for u in users.values())

                    # 运行所有算法
                    algorithm_results = {}

                    # 1-split-augment作为baseline
                    _, _, baseline_cost, _ = run_algorithm(network, users, llms,
                                                           '1-split-augment', 1)

                    # no-split
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   'no-split')
                    algorithm_results['no-split'] = (cost, sr)

                    # 1-split
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   '1-split', 1)
                    algorithm_results['1-split'] = (cost, sr)

                    # 100-split-augment
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   '100-split-augment', 100)
                    algorithm_results['100-split-augment'] = (cost, sr)

                    # bottleneck-augment
                    _, _, cost, sr = run_algorithm(network, users, llms,
                                                   'bottleneck-augment')
                    algorithm_results['bottleneck-augment'] = (cost, sr)

                    # 计算optimization（相对于1-split-augment）
                    for alg, (cost, sr) in algorithm_results.items():
                        if alg not in data_by_algorithm:
                            data_by_algorithm[alg] = []

                        if baseline_cost > 0 and cost > 0:
                            opt_percentage = (cost - baseline_cost) / cost * 100
                        else:
                            opt_percentage = 0

                        data_by_algorithm[alg].append((opt_percentage, sr,
                                                       cost))

                        # 记录到Excel行（累积到总列表）
                        all_excel_rows.append({
                            'llm_computation':
                            computation,
                            'bandwidth':
                            bandwidth,
                            'algorithm':
                            alg,
                            'total_cost':
                            cost,
                            'service_rate':
                            sr,
                            'optimization_vs_1split_augment':
                            opt_percentage
                        })

                # 绘制图表
                plot_filename = os.path.join(
                    _LLM_DIR, f'{distribution_name}_comp{computation}.png')
                plot_dual_axis(
                    BANDWIDTH_VALUES, data_by_algorithm, 'Bandwidth',
                    f'{distribution_name} - LLM Computation={computation}',
                    plot_filename)

            # 将该分布的数据添加到总字典
            llm_all_distributions[distribution_name] = all_excel_rows

    # 循环2结束后，保存所有分布到一个Excel文件
    llm_excel_file = os.path.join(_LLM_DIR, 'llm.xlsx')
    save_to_excel_multi_sheet(llm_all_distributions, llm_excel_file)


if __name__ == "__main__":
    main()
