import Entity
import pandas as pd
import numpy as np
import math
import time
import networkx as nx
import os
import copy
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

DISTRIBUTION_TYPES = Entity.DISTRIBUTION_TYPES
script_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(script_path, 'results')
is_shared = 1

_RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(_RESULT_DIR, exist_ok=True)
origin_file = os.path.join(_RESULT_DIR, 'OriginResults.xlsx')
dash_file = os.path.join(_RESULT_DIR, 'DashBoard.xlsx')
link_util_file = os.path.join(_RESULT_DIR, 'LinkUtilization.xlsx')
runtime_file = os.path.join(_RESULT_DIR, 'runtime.xlsx')
bottleneck_k_file = os.path.join(_RESULT_DIR, 'bottleneck-k.xlsx')
reverse_edge_saving_file = os.path.join(_RESULT_DIR,
                                        'reverse-edge-cost-saving.xlsx')

_BACKDRAW_DIR = os.path.join(os.path.dirname(__file__), 'backdraw')
os.makedirs(_BACKDRAW_DIR, exist_ok=True)

_WITHDRAW_DIR = os.path.join(os.path.dirname(__file__), 'withdraw')
os.makedirs(_WITHDRAW_DIR, exist_ok=True)


def plot_bottleneck_withdraw_cost(distribution_name, backtrack_iters,
                                   cost_comparisons):
    """
    为bottleneck-augment绘制回撤成本优化图

    参数:
        distribution_name: 分布名称
        backtrack_iters: 使用反向边的迭代序号列表
        cost_comparisons: 成本对比列表 (augment_cost, greedy_cost, diff)
    """

    if not backtrack_iters:
        print(f"    bottleneck-augment: 无反向边使用，跳过")
        return

    # 分离有解和无解的情况
    valid_x = []
    valid_y = []
    invalid_x = []

    for idx, comp in enumerate(cost_comparisons, start=1):
        if comp[2] != float('inf'):
            valid_x.append(idx)
            valid_y.append(comp[2])
        else:
            invalid_x.append(idx)

    if not valid_x and not invalid_x:
        print(f"    bottleneck-augment: 无有效数据，跳过")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制有解的部分
    if valid_x:
        ax.plot(valid_x,
                valid_y,
                marker='o',
                color='tab:purple',
                label='bottleneck-augment',
                linewidth=2,
                markersize=6,
                alpha=0.8)

    # 绘制无解的点（红色X标记在y=0位置）
    if invalid_x:
        ax.scatter(invalid_x,
                   [0] * len(invalid_x),
                   marker='x',
                   color='red',
                   s=100,
                   label='greedy no solution',
                   zorder=5)

    # 添加零线参考
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('withdraw_order', fontsize=13)
    ax.set_ylabel('potimization', fontsize=13)
    ax.set_title(f'{distribution_name} (bottleneck)',
                 fontsize=15,
                 fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # 添加说明文本
    note_text = 'comparison: greedy_cost - augment_cost'
    ax.text(0.02,
            0.98,
            note_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    # 保存图片
    filename = os.path.join(_WITHDRAW_DIR,
                            f'{distribution_name}-bottleneck.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    已保存bottleneck回撤对比图: {filename}")


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
                              original_order=('original_order', 'min'),
                          ).reset_index())

    # 按照原始分配顺序排序，保持真实的算法分配顺序
    grouped = grouped.sort_values('original_order').reset_index(drop=True)

    return grouped


def write_origin_results(file_path, blocks, user_distribution,
                         llm_distribution):
    """
    将所有算法结果写入同一个 sheet，每个算法之间空一行。
    使用追加模式，保留已有的其他 sheet。
    """
    sheet_name = f'{user_distribution}-{llm_distribution}'

    # 先将所有 blocks 合并成一个大 DataFrame
    all_rows = []
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

        # 添加数据行和总计行
        df_with_total = pd.concat(
            [df_to_write, pd.DataFrame([total_row])], ignore_index=True)
        all_rows.append(df_with_total)

        # 添加空行（用空字典表示）
        empty_row = pd.DataFrame([{col: '' for col in df_with_total.columns}])
        all_rows.append(empty_row)

    # 合并所有 blocks（移除最后一个空行）
    if all_rows:
        if len(all_rows) > 1:
            all_rows.pop()  # 移除最后的空行
        final_df = pd.concat(all_rows, ignore_index=True)
    else:
        # 如果没有数据，创建空 DataFrame
        final_df = pd.DataFrame(columns=[
            'algorithm', 'user_id', 'llm_id', 'path', 'total_flow',
            'total_cost', 'original_order'
        ])

    # 追加模式：如果文件存在则追加新 sheet，否则创建新文件
    mode = 'a' if os.path.exists(file_path) else 'w'
    kwargs = {'mode': mode, 'engine': 'openpyxl'}
    if mode == 'a':
        kwargs['if_sheet_exists'] = 'replace'  # 如果 sheet 已存在则替换

    # 一次性写入整个 sheet
    with pd.ExcelWriter(file_path, **kwargs) as writer:
        final_df.to_excel(writer, sheet_name=sheet_name, index=False)


def compute_edge_betweenness(alo_network, users, llms, top_n=10):
    """
    返回按介数中心性排序的前 top_n 条有向链路。

    参数:
        alo_network: 网络对象
        users: 用户字典 {user_id: User}
        llms: LLM字典 {llm_id: LLM}
        top_n: 返回前 N 条边

    注意: 只计算从 user 节点到 LLM 节点的最短路径的介数中心性，
         剔除无关路径，更准确地反映业务流量的关键链路。
    """
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

    # 提取 user 和 LLM 节点 ID
    user_nodes = list(users.keys())
    llm_nodes = list(llms.keys())

    # 手动计算受限介数中心性：只考虑 user → LLM 路径
    edge_count = {}
    total_paths = 0

    for user_id in user_nodes:
        for llm_id in llm_nodes:
            try:
                # 计算加权最短路径
                path = nx.shortest_path(G,
                                        source=user_id,
                                        target=llm_id,
                                        weight='weight')
                total_paths += 1

                # 统计路径中的每条边
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            except nx.NetworkXNoPath:
                # 如果没有路径，跳过
                continue

    # 归一化：除以总路径数
    edge_centrality = {}
    if total_paths > 0:
        for edge, count in edge_count.items():
            edge_centrality[edge] = count / total_paths

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
                                  top_edges):
    """
    保存链路利用率报告：
    - 第一列为算法名
    - 第二列为 top-k 链路的平均利用率
    - 后续列为同一组前 top_n 关键链路的利用率
    - 末尾附带每条关键链路的端点与介数中心性
    """
    edge_pairs = [edge for edge, _ in top_edges]
    edge_betweenness = [bet for _, bet in top_edges]

    columns = ['algorithm', 'avg_utilization_topk']

    rows = []
    for alg_name, util_map in algorithm_utils:
        row = {col: '' for col in columns}
        row['algorithm'] = alg_name

        # 计算 top-k 链路的平均利用率
        utilizations = []
        for idx, ((src, dst), bet) in enumerate(zip(edge_pairs,
                                                    edge_betweenness),
                                                start=1):
            columns.append(f'{src}->{dst} utilization')
            util_key = f'{src}->{dst} utilization'

            util_value = util_map.get((src, dst), 0.0)
            row[util_key] = util_value

            utilizations.append(util_value)

        # 平均利用率
        row['avg_utilization_topk'] = sum(utilizations) / len(
            utilizations) if utilizations else 0.0

        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    # 追加模式：如果文件存在则追加新 sheet，否则创建新文件
    mode = 'a' if os.path.exists(file_path) else 'w'
    kwargs = {'mode': mode, 'engine': 'openpyxl'}
    if mode == 'a':
        kwargs['if_sheet_exists'] = 'replace'

    with pd.ExcelWriter(file_path, **kwargs) as writer:
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

    # 追加模式：如果文件存在则追加新 sheet，否则创建新文件
    mode = 'a' if os.path.exists(file_path) else 'w'
    kwargs = {'mode': mode, 'engine': 'openpyxl'}
    if mode == 'a':
        kwargs['if_sheet_exists'] = 'replace'

    with pd.ExcelWriter(file_path, **kwargs) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def write_bottleneck_k_report(file_path, sheet_name, allocations):
    """
    保存 bottleneck-augment 每轮推流的瓶颈流量统计

    参数:
        file_path: Excel 文件路径
        sheet_name: sheet 名称（格式：user_distribution-llm_distribution）
        allocations: bottleneck-augment 的分配结果列表
    """
    if not allocations:
        return

    rows = []
    for idx, alloc in enumerate(allocations):
        path_str = '->'.join(map(str, alloc['path']))
        rows.append({
            'iteration': idx + 1,
            'k_value': alloc['flow'],
            'user_id': alloc['user_id'],
            'llm_id': alloc['llm_id'],
            'path': path_str,
            'cost': alloc['cost']
        })

    df = pd.DataFrame(rows)

    # 添加统计信息
    k_values = [alloc['flow'] for alloc in allocations]
    stats_rows = [{
        'iteration': 'Total',
        'k_value': sum(k_values),
        'user_id': '',
        'llm_id': '',
        'path': f'{len(allocations)} iterations',
        'cost': sum(alloc["cost"] for alloc in allocations)
    }, {
        'iteration': 'Mean',
        'k_value': sum(k_values) / len(k_values) if k_values else 0,
        'user_id': '',
        'llm_id': '',
        'path': '',
        'cost': ''
    }, {
        'iteration': 'Min',
        'k_value': min(k_values) if k_values else 0,
        'user_id': '',
        'llm_id': '',
        'path': '',
        'cost': ''
    }, {
        'iteration': 'Max',
        'k_value': max(k_values) if k_values else 0,
        'user_id': '',
        'llm_id': '',
        'path': '',
        'cost': ''
    }]

    df = pd.concat([df, pd.DataFrame(stats_rows)], ignore_index=True)

    # 追加模式：如果文件存在则追加新 sheet，否则创建新文件
    mode = 'a' if os.path.exists(file_path) else 'w'
    kwargs = {'mode': mode, 'engine': 'openpyxl'}
    if mode == 'a':
        kwargs['if_sheet_exists'] = 'replace'

    with pd.ExcelWriter(file_path, **kwargs) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def write_reverse_edge_saving_report(file_path, sheet_name, allocations):
    """
    保存增广路径算法中反向边带来的成本优化统计

    参数:
        file_path: Excel 文件路径
        sheet_name: sheet 名称（格式：algorithm-distribution）
        allocations: 增广算法的分配结果列表
    """
    if not allocations:
        return

    # 只统计使用了反向边的路径
    reverse_edge_paths = [
        alloc for alloc in allocations
        if alloc.get('reverse_edge_count', 0) > 0
    ]

    if not reverse_edge_paths:
        # 如果没有使用反向边，创建空报告
        rows = [{
            'iteration': 'N/A',
            'reverse_edge_count': 0,
            'cost_saving': 0,
            'flow': 0,
            'user_id': '',
            'llm_id': '',
            'path': 'No reverse edges used',
            'total_cost': 0
        }]
    else:
        rows = []
        for idx, alloc in enumerate(reverse_edge_paths):
            path_str = '->'.join(map(str, alloc['path']))
            rows.append({
                'iteration':
                idx + 1,
                'reverse_edge_count':
                alloc.get('reverse_edge_count', 0),
                'cost_saving':
                alloc.get('reverse_edge_cost_saving', 0),
                'flow':
                alloc['flow'],
                'user_id':
                alloc['user_id'],
                'llm_id':
                alloc['llm_id'],
                'path':
                path_str,
                'total_cost':
                alloc['cost']
            })

    df = pd.DataFrame(rows)

    # 添加统计信息
    if reverse_edge_paths:
        total_saving = sum(
            alloc.get('reverse_edge_cost_saving', 0)
            for alloc in reverse_edge_paths)
        total_reverse_edges = sum(
            alloc.get('reverse_edge_count', 0) for alloc in reverse_edge_paths)
        stats_rows = [{
            'iteration':
            'Total',
            'reverse_edge_count':
            total_reverse_edges,
            'cost_saving':
            total_saving,
            'flow':
            sum(alloc['flow'] for alloc in reverse_edge_paths),
            'user_id':
            '',
            'llm_id':
            '',
            'path':
            f'{len(reverse_edge_paths)} paths with reverse edges',
            'total_cost':
            sum(alloc['cost'] for alloc in reverse_edge_paths)
        }, {
            'iteration':
            'Mean Saving',
            'reverse_edge_count':
            '',
            'cost_saving':
            total_saving /
            len(reverse_edge_paths) if reverse_edge_paths else 0,
            'flow':
            '',
            'user_id':
            '',
            'llm_id':
            '',
            'path':
            '',
            'total_cost':
            ''
        }]
        df = pd.concat([df, pd.DataFrame(stats_rows)], ignore_index=True)

    # 追加模式：如果文件存在则追加新 sheet，否则创建新文件
    mode = 'a' if os.path.exists(file_path) else 'w'
    kwargs = {'mode': mode, 'engine': 'openpyxl'}
    if mode == 'a':
        kwargs['if_sheet_exists'] = 'replace'

    with pd.ExcelWriter(file_path, **kwargs) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def plot_reverse_edge_cost_saving(allocations, algorithm_name, distribution_name):
    """
    绘制反向边成本优化图表

    参数:
        allocations: 增广算法的分配结果列表
        algorithm_name: 算法名称
        distribution_name: 分布名称
    """
    if not allocations:
        return

    # 只统计使用了反向边的路径
    reverse_edge_paths = [
        alloc for alloc in allocations
        if alloc.get('reverse_edge_count', 0) > 0
    ]

    if not reverse_edge_paths:
        print(f"  {algorithm_name}-{distribution_name}: 无反向边使用，跳过绘图")
        return

    # 提取数据
    iterations = list(range(1, len(reverse_edge_paths) + 1))
    cost_savings = [alloc.get('reverse_edge_cost_saving', 0) for alloc in reverse_edge_paths]
    reverse_edge_counts = [alloc.get('reverse_edge_count', 0) for alloc in reverse_edge_paths]
    cumulative_savings = np.cumsum(cost_savings)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 上图：每次迭代的成本节省（柱状图）+ 累计节省（折线图）
    color1 = 'tab:blue'
    ax1.bar(iterations, cost_savings, color=color1, alpha=0.6, label='单次成本节省')
    ax1.set_xlabel('使用反向边的迭代序号', fontsize=12)
    ax1.set_ylabel('单次成本节省', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 添加累计节省折线
    ax1_twin = ax1.twinx()
    color2 = 'tab:red'
    ax1_twin.plot(iterations, cumulative_savings, color=color2, marker='o',
                  linewidth=2, markersize=4, label='累计成本节省')
    ax1_twin.set_ylabel('累计成本节省', color=color2, fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title(f'{algorithm_name} - {distribution_name}\n反向边成本优化统计',
                  fontsize=14, fontweight='bold')

    # 下图：每次迭代的反向边数量
    ax2.bar(iterations, reverse_edge_counts, color='tab:green', alpha=0.6)
    ax2.set_xlabel('使用反向边的迭代序号', fontsize=12)
    ax2.set_ylabel('反向边数量', fontsize=12)
    ax2.set_title('每次迭代使用的反向边数量', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 添加统计信息文本
    total_saving = sum(cost_savings)
    total_edges = sum(reverse_edge_counts)
    avg_saving = total_saving / len(cost_savings) if cost_savings else 0

    stats_text = f'总节省: {total_saving:.1f}\n使用反向边次数: {len(reverse_edge_paths)}\n平均单次节省: {avg_saving:.1f}\n反向边总数: {total_edges}'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存图片
    filename = os.path.join(_BACKDRAW_DIR, f'{algorithm_name}-{distribution_name}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  反向边成本优化图已保存: {filename}")


def add_cost_comparison_column(file_path):
    """
    添加相较于同分布下最小k的k-split-augment算法的花销百分比列

    公式：(此算法花销 - 基准算法花销) / 此算法花销
    自动查找每个分布下k值最小的k-split-augment作为基准
    """
    if not os.path.exists(file_path):
        return

    df = pd.read_excel(file_path)

    # 如果列已存在，先删除
    if 'cost_vs_baseline_augment' in df.columns:
        df = df.drop(columns=['cost_vs_baseline_augment'])

    # 为每个分布找到最小k的augment算法作为基准
    distribution_baselines = {}
    for dist in df['distribution'].unique():
        dist_df = df[df['distribution'] == dist]
        # 查找所有 k-split-augment-* 算法
        augment_rows = dist_df[dist_df['algorithm'].str.startswith(
            'k-split-augment-', na=False)]

        if not augment_rows.empty:
            # 提取k值并找到最小的
            k_values = []
            for alg in augment_rows['algorithm']:
                try:
                    k = int(alg.split('-')[-1])
                    k_values.append(k)
                except:
                    continue

            if k_values:
                min_k = min(k_values)
                baseline_alg = f'k-split-augment-{min_k}'
                baseline_cost = dist_df[dist_df['algorithm'] ==
                                        baseline_alg]['total_cost'].iloc[0]
                distribution_baselines[dist] = (baseline_alg, baseline_cost)

    # 为每一行计算百分比
    cost_comparison = []
    for _, row in df.iterrows():
        dist = row['distribution']
        this_cost = row['total_cost']

        if dist not in distribution_baselines or this_cost == 0:
            cost_comparison.append(None)
        else:
            baseline_alg, baseline_cost = distribution_baselines[dist]
            percentage = (this_cost - baseline_cost) / this_cost
            cost_comparison.append(percentage)

    # 插入到 total_cost 列之后
    cost_col_idx = df.columns.get_loc('total_cost') + 1
    df.insert(cost_col_idx, 'cost_vs_baseline_augment', cost_comparison)

    df.to_excel(file_path, index=False)

    # 输出使用的基准信息
    print("Baseline algorithms used for each distribution:")
    for dist, (alg, cost) in distribution_baselines.items():
        print(f"  {dist}: {alg} (cost={cost:.2f})")


def write_dashboard_report(file_path,
                           distribution_name,
                           algorithm_name,
                           network,
                           allocations,
                           users,
                           reverse_edge_flow=0,
                           runtime=0,
                           top15_utilization=0):
    """
    统计并保存算法性能指标到dashboard

    参数:
        file_path: Excel 文件路径
        distribution_name: 分布名称（user_dist-llm_dist）
        algorithm_name: 算法名称
        network: Network对象
        allocations: 分配结果列表
        users: 用户字典（用于计算服务率）
        reverse_edge_flow: 反向边流量（默认0，只有augment算法需要传入）
        runtime: 运行时间（秒）
        top15_utilization: 前15条关键链路的平均利用率
    """
    if not allocations:
        return

    # 计算总需求
    total_demand = sum(u.bw for u in users.values())

    # 计算实际服务流量
    total_served = sum(alloc['flow'] for alloc in allocations)

    # 服务率
    service_rate = total_served / total_demand if total_demand > 0 else 0

    # 总花销
    total_cost = sum(alloc['cost'] for alloc in allocations)

    # 统计路径指标（从网络最终状态统计，而非从allocation）
    # 这样可以正确反映反向边对流量分配的影响
    total_flow_distance = 0
    total_flow_hops = 0

    for link_list in network.links.values():
        for link in link_list:
            # 只统计正向边且有流量的边
            if not link.is_reverse and link.flow > 1e-9:
                total_flow_distance += link.flow * link.distance
                total_flow_hops += link.flow  # 每单位流量经过1跳

    # 计算实际服务流量
    total_served = sum(alloc['flow'] for alloc in allocations)

    # 计算加权平均值
    avg_physical_distance = total_flow_distance / total_served if total_served > 0 else 0
    avg_hops = total_flow_hops / total_served if total_served > 0 else 0

    # 构建数据行
    row = {
        'distribution': distribution_name,
        'algorithm': algorithm_name,
        'service_rate': service_rate,
        'total_cost': total_cost,
        'runtime_seconds': runtime,
        'top15_avg_utilization': top15_utilization,
        'avg_path_length': avg_physical_distance,
        'avg_hops': avg_hops,
        'reverse_edge_flow': reverse_edge_flow
    }

    # 追加模式写入Excel
    mode = 'a' if os.path.exists(file_path) else 'w'

    if mode == 'a':
        # 读取现有数据并追加
        existing_df = pd.read_excel(file_path)
        new_df = pd.concat([existing_df, pd.DataFrame([row])],
                           ignore_index=True)
    else:
        new_df = pd.DataFrame([row])

    new_df.to_excel(file_path, index=False)


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
            network_copy.add_one_way_link(lid, T, max_customers * single_bw, 1e-19)
    else:
        for lid, llm in llms.items():
            network_copy.add_one_way_link(lid, T, total_bw, 1e-19)

    allocations = []
    remaining = total_bw
    while remaining >= 1e-9:
        push = 500
        dist, prev = network_copy.dijkstra_with_capacity(S,
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

        # 计算物理距离和跳数（排除超源和超汇的边）
        physical_distance = sum(lk.distance for lk in link_path[1:-1])
        hops = len(link_path) - 2

        allocations.append({
            "algorithm": "no_split",
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': dist[T] * push,
            'flow': push,
            'physical_distance': physical_distance,
            'hops': hops
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
            network_copy.add_one_way_link(lid, T, max_customers * single_bw, 1e-19)
    else:
        for lid, llm in llms.items():
            network_copy.add_one_way_link(lid, T, total_bw, 1e-19)

    allocations = []
    remaining = total_bw
    while remaining >= 1e-9:
        push = min(k, remaining)
        dist, prev = network_copy.dijkstra_with_capacity(S,
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

        # 计算物理距离和跳数（排除超源和超汇的边）
        physical_distance = sum(lk.distance for lk in link_path[1:-1])
        hops = len(link_path) - 2

        allocations.append({
            "algorithm": f"{k}-split",
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': dist[T] * push,
            'flow': push,
            'physical_distance': physical_distance,
            'hops': hops
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
            net.add_one_way_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid, llm in llms.items():
            net.add_one_way_link(lid, T, total_bw, 0)

    # 运行最小费用流，获取反向边统计
    allocations, reverse_edge_count, _, _ = net.successive_shortest_paths(
        S, T, total_bw, k=k)

    # 基于最终流量结果扣减 LLM 计算资源
    if is_shared and single_bw > 0:
        for allocation in allocations:
            llm_id = allocation['llm_id']
            flow_units = allocation['flow'] / single_bw
            llms[llm_id].available_computation -= single_cpu * flow_units

    return allocations, net, reverse_edge_count


def bottleneck_augment(network, users, llms):
    print("进入了 bottleneck-augment 算法")
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
            net.add_one_way_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid, llm in llms.items():
            net.add_one_way_link(lid, T, total_bw, 0)

    # 运行最小费用流，使用瓶颈流量
    allocations, reverse_edge_flow, backtrack_iters, cost_comparisons = net.successive_shortest_paths(
        S, T, total_bw, k=1, use_bottleneck=True)

    # 基于最终流量结果扣减 LLM 计算资源
    if is_shared and single_bw > 0:
        for allocation in allocations:
            llm_id = allocation['llm_id']
            flow_units = allocation['flow'] / single_bw
            llms[llm_id].available_computation -= single_cpu * flow_units

    return allocations, net, reverse_edge_flow, backtrack_iters, cost_comparisons


if __name__ == "__main__":
    for user_distribution in DISTRIBUTION_TYPES:
        for llm_distribution in DISTRIBUTION_TYPES:
            # user_distribution = 'uniform'
            # llm_distribution = 'uniform'

            # 先加载LLM信息，然后加载网络（带LLM标识）
            llms = Entity.load_llm_info(user_distribution, llm_distribution)
            json = Entity.load_network_from_sheets(llm_ids=llms.keys())
            network = json['network']
            nodes_list = list(json['nodes'].values())
            nodes = json['nodes']
            users = Entity.load_user_info(user_distribution)
            for llm in llms.values():
                nodes_list[llm.id].role = 'llm'
                nodes_list[llm.id].deployed = 1
            for user in users.values():
                nodes_list[user.id].role = 'user'

            # 不再需要convert_to_llm_endpoints，因为网络加载时已经处理

            # 用户按带宽排序
            users = dict(
                sorted(users.items(),
                       key=lambda item: item[1].bw,
                       reverse=True))

            user_ideal_llms = {}
            for user in users.values():
                distances, costs = network.dijkstra_ideal(user.id, user.bw)
                sorted_nodes = sorted(distances, key=distances.get)
                ideal_llms = {
                    n: costs[n]
                    for n in sorted_nodes if nodes_list[n].role == 'llm'
                }
                user_ideal_llms[user.id] = ideal_llms

            Entity.visualize_network(nodes_list, network, llms, users,
                                     user_distribution, llm_distribution)

            # 初始化运行时间记录字典
            runtime_data = {}

            # no_split
            start_time = time.time()
            no_split_allocations, no_split_network = no_split(
                network, users, llms, user_ideal_llms)
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
            augment_reverse_flows = {kid: 0 for kid in ks}  # 存储反向边流量
            distribution_name = f'{user_distribution}-{llm_distribution}'

            for split_k in ks:
                start_time = time.time()
                augment_allocations, augment_network, reverse_flow = k_split_augment(
                    network, users, llms, k=split_k)
                runtime_data[f'k-split-augment-{split_k}'] = time.time(
                ) - start_time
                augment_results[split_k] = augment_allocations
                augment_networks[split_k] = augment_network
                augment_reverse_flows[split_k] = reverse_flow  # 保存反向边流量

                network.reset_network(llms)

            # bottleneck_augment
            start_time = time.time()
            (bottleneck_allocations, bottleneck_network, bottleneck_reverse_flow,
             bottleneck_backtrack_iters,
             bottleneck_cost_comparisons) = bottleneck_augment(
                 network, users, llms)
            runtime_data['bottleneck-augment'] = time.time() - start_time
            network.reset_network(llms)

            # 绘制bottleneck回撤成本优化图
            distribution_name = f'{user_distribution}-{llm_distribution}'
            plot_bottleneck_withdraw_cost(distribution_name,
                                          bottleneck_backtrack_iters,
                                          bottleneck_cost_comparisons)

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

            # bottleneck_augment 结果
            if bottleneck_allocations:
                bottleneck_df = merge_allocations_by_path(
                    bottleneck_allocations)
                all_blocks.append(('bottleneck-augment', bottleneck_df))

            write_origin_results(origin_file, all_blocks, user_distribution,
                                 llm_distribution)

            # 保存 bottleneck-augment 每轮推流的 k 值统计
            if bottleneck_allocations:
                write_bottleneck_k_report(
                    bottleneck_k_file,
                    f'{user_distribution}-{llm_distribution}',
                    bottleneck_allocations)
                print(f"Bottleneck-k report saved to {bottleneck_k_file}")

            # Link utilization report
            # 使用受限介数中心性：只计算 user→LLM 路径
            top_edges = compute_edge_betweenness(network,
                                                 users,
                                                 llms,
                                                 top_n=15)
            critical_edges = [edge for edge, _ in top_edges]

            algorithm_networks = [('no-split', no_split_network)]
            algorithm_networks.extend([
                (f'k-split-{k_val}', net)
                for k_val, net in k_split_networks.items() if net is not None
            ])
            algorithm_networks.extend([
                (f'k-split-augment-{k_val}', net)
                for k_val, net in augment_networks.items() if net is not None
            ])
            # 添加 bottleneck_augment
            algorithm_networks.append(
                ('bottleneck-augment', bottleneck_network))

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

            # 保存dashboard统计报告
            distribution_name = f'{user_distribution}-{llm_distribution}'

            # 将algorithm_utils转为字典，方便查询
            util_dict = {
                alg_name: util_map
                for alg_name, util_map in algorithm_utils
            }

            # no-split算法
            top15_util = 0
            if 'no-split' in util_dict:
                utils = list(util_dict['no-split'].values())
                top15_util = sum(utils) / len(utils) if utils else 0

            write_dashboard_report(dash_file, distribution_name, 'no-split',
                                   no_split_network,
                                   no_split_allocations, users, 0,
                                   runtime_data.get('no-split', 0), top15_util)

            # k-split算法
            for k_val, allocs in k_split_results.items():
                if allocs:
                    alg_name = f'k-split-{k_val}'
                    top15_util = 0
                    if alg_name in util_dict:
                        utils = list(util_dict[alg_name].values())
                        top15_util = sum(utils) / len(utils) if utils else 0

                    write_dashboard_report(dash_file, distribution_name,
                                           alg_name, k_split_networks[k_val],
                                           allocs, users, 0,
                                           runtime_data.get(alg_name,
                                                            0), top15_util)

            # k-split-augment算法
            for k_val, allocs in augment_results.items():
                if allocs:
                    alg_name = f'k-split-augment-{k_val}'
                    top15_util = 0
                    if alg_name in util_dict:
                        utils = list(util_dict[alg_name].values())
                        top15_util = sum(utils) / len(utils) if utils else 0

                    write_dashboard_report(dash_file, distribution_name,
                                           alg_name, augment_networks[k_val],
                                           allocs, users,
                                           augment_reverse_flows[k_val],
                                           runtime_data.get(alg_name,
                                                            0), top15_util)

            # bottleneck-augment算法
            if bottleneck_allocations:
                top15_util = 0
                if 'bottleneck-augment' in util_dict:
                    utils = list(util_dict['bottleneck-augment'].values())
                    top15_util = sum(utils) / len(utils) if utils else 0

                write_dashboard_report(
                    dash_file, distribution_name, 'bottleneck-augment',
                    bottleneck_network, bottleneck_allocations, users,
                    bottleneck_reverse_flow,
                    runtime_data.get('bottleneck-augment', 0), top15_util)

            # 保存反向边成本优化报告
            # k-split-augment 算法
            for k_val, allocs in augment_results.items():
                if allocs:
                    alg_name = f'k-split-augment-{k_val}'
                    write_reverse_edge_saving_report(
                        reverse_edge_saving_file,
                        f'{alg_name}-{distribution_name}', allocs)
                    # 绘制反向边成本优化图
                    plot_reverse_edge_cost_saving(allocs, alg_name, distribution_name)

            # bottleneck-augment 算法
            if bottleneck_allocations:
                write_reverse_edge_saving_report(
                    reverse_edge_saving_file,
                    f'bottleneck-augment-{distribution_name}',
                    bottleneck_allocations)
                # 绘制反向边成本优化图
                plot_reverse_edge_cost_saving(bottleneck_allocations, 'bottleneck-augment', distribution_name)

            print(f"Dashboard report saved to {dash_file}")
            print(
                f"Reverse edge cost saving report saved to {reverse_edge_saving_file}"
            )

    # 所有分布组合的算法运行完毕后，添加花销对比列
    add_cost_comparison_column(dash_file)
    print(f"Cost comparison column added to {dash_file}")
