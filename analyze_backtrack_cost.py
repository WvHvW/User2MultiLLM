"""
分析增广路径算法中回撤流量带来的成本优化

对比同粒度下 k-split（贪心）vs k-split-augment（增广）的单次推流成本差异
只统计使用反向边的轮次
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
K_VALUES = [1, 10, 100]  # 测试的粒度值

# 输出目录
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'withdraw')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def k_split_with_trace(network, users, llms, k):
    """
    贪心k-split算法，记录每次推流后的累计成本

    返回:
        allocations: 分配结果
        cost_trace: 每次推流后的累计成本列表
    """
    net = copy.deepcopy(network)

    u0 = next(iter(users.values()))
    single_bw = u0.bw
    single_cpu = u0.computation
    total_bw = sum(u.bw for u in users.values())

    S, T = -1, -2
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    for uid, u in users.items():
        net.add_link(S, uid, u.bw, 0)

    if is_shared:
        for lid, llm in llms.items():
            max_customers = int(llm.computation // single_cpu)
            net.add_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid, llm in llms.items():
            net.add_link(lid, T, total_bw, 0)

    allocations = []
    cost_trace = []  # 记录累计成本
    cumulative_cost = 0
    remaining = total_bw

    while remaining > 1e-9:
        push = min(k, remaining)

        # Dijkstra 找最短路径（只考虑正向边）
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
            llms[llm_id].available_computation -= single_cpu

        path_cost = dist[T] * push
        cumulative_cost += path_cost
        cost_trace.append(cumulative_cost)

        allocations.append({
            "algorithm": f"k-split-{k}",
            'user_id': user_id,
            'llm_id': llm_id,
            'path': node_path[1:-1],
            'cost': path_cost,
            'flow': push
        })
        remaining -= push

    return allocations, cost_trace, net


def k_split_augment_with_comparison(network, users, llms, k):
    """
    增广路径算法，在每次使用反向边时，对比贪心算法的路径成本

    返回:
        backtrack_iters: 使用反向边的迭代序号列表
        cost_comparisons: 每次使用反向边时的成本对比 (augment_cost, greedy_cost, diff)
    """
    net = copy.deepcopy(network)

    u0 = next(iter(users.values()))
    single_bw = u0.bw
    single_cpu = u0.computation
    total_bw = sum(u.bw for u in users.values())

    S, T = -1, -2
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    for uid, u in users.items():
        net.add_link(S, uid, u.bw, 0)

    if is_shared:
        for lid, llm in llms.items():
            max_customers = int(llm.computation // single_cpu)
            net.add_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid, llm in llms.items():
            net.add_link(lid, T, total_bw, 0)

    import math
    from collections import deque
    INF = math.inf
    EPSILON = 1e-9

    backtrack_iters = []
    cost_comparisons = []
    remaining = total_bw
    iteration = 0

    while remaining > 1e-9:
        iteration += 1
        push = min(k, remaining)

        # 运行 SPFA 找最短路径（允许使用反向边）
        dist = {nid: INF for nid in net.nodes}
        prev = {nid: None for nid in net.nodes}
        in_queue = {nid: False for nid in net.nodes}

        dist[S] = 0
        queue = deque([S])
        in_queue[S] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            for link in net.links.get(u, []):
                rc = link.residual_capacity
                if rc < push:
                    continue

                v = link.dst
                cost = link.distance if not link.is_reverse else -link.reverse.distance

                if dist[u] + cost < dist[v] - EPSILON:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, link)

                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        if dist[T] == INF:
            break

        # 提取增广路径
        node_path = []
        link_path = []
        cur = T
        while cur != S:
            node_path.append(cur)
            prev_node, prev_link = prev[cur]
            link_path.append(prev_link)
            cur = prev_node
        node_path.append(S)
        node_path.reverse()
        link_path.reverse()

        augment_path_cost = dist[T] * push

        # 检查是否使用了反向边（排除超源超汇的边）
        used_reverse = False
        for i, lk in enumerate(link_path):
            if i == 0 or i == len(link_path) - 1:
                continue
            if lk.is_reverse:
                used_reverse = True
                break

        # 如果使用了反向边，计算贪心路径的成本
        if used_reverse:
            # 在当前网络状态下运行 Dijkstra（只考虑正向边）
            greedy_dist, greedy_prev = net.dijkstra_with_capacity(S, push)

            if greedy_dist[T] < INF:
                greedy_path_cost = greedy_dist[T] * push
                cost_diff = greedy_path_cost - augment_path_cost

                backtrack_iters.append(iteration)
                cost_comparisons.append(
                    (augment_path_cost, greedy_path_cost, cost_diff))
            else:
                # 贪心无法找到路径
                backtrack_iters.append(iteration)
                cost_comparisons.append((augment_path_cost, INF, INF))

        # 推流
        net.send_flow(link_path, push)
        if is_shared:
            user_id = node_path[1]
            llm_id = node_path[-2]
            llms[llm_id].available_computation -= single_cpu

        remaining -= push

    return backtrack_iters, cost_comparisons, net


def plot_withdraw_cost_single_k(distribution_name, k, backtrack_iters,
                                cost_comparisons):
    """
    为单个k值绘制回撤成本优化图

    参数:
        distribution_name: 分布名称
        k: 粒度值
        backtrack_iters: 使用反向边的迭代序号列表
        cost_comparisons: 成本对比列表 (augment_cost, greedy_cost, diff)
    """

    if not backtrack_iters:
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
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制有解的部分
    if valid_x:
        ax.plot(valid_x,
                valid_y,
                marker='o',
                color='tab:blue',
                label=f'k-split-augment-{k}',
                linewidth=2,
                markersize=6,
                alpha=0.8)

    # 绘制无解的点（红色X标记在y=0位置）
    if invalid_x:
        ax.scatter(invalid_x, [0] * len(invalid_x),
                   marker='x',
                   color='red',
                   s=100,
                   label='greedy no solution',
                   zorder=5)

    # 添加零线参考
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('withdraw_order', fontsize=13)
    ax.set_ylabel('potimization', fontsize=13)
    ax.set_title(f'{distribution_name} (k={k})',
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
    filename = os.path.join(_OUTPUT_DIR, f'{distribution_name}-k{k}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    已保存: {filename}")


def main():
    for user_distribution in DISTRIBUTION_TYPES:
        for llm_distribution in DISTRIBUTION_TYPES:
            distribution_name = f'{user_distribution}-{llm_distribution}'

            # 加载网络数据
            json = Entity.load_network_from_sheets()
            network = json['network']
            nodes_list = list(json['nodes'].values())
            llms = Entity.load_llm_info(user_distribution, llm_distribution)
            users = Entity.load_user_info(user_distribution)

            for llm in llms.values():
                nodes_list[llm.id].role = 'llm'
                nodes_list[llm.id].deployed = 1
            for user in users.values():
                nodes_list[user.id].role = 'user'

            # 用户按带宽排序
            users = dict(
                sorted(users.items(),
                       key=lambda item: item[1].bw,
                       reverse=True))

            # 对每个k值进行分析并单独绘图
            for k in K_VALUES:

                # 运行增广算法并对比贪心路径
                backtrack_iters, cost_comparisons, augment_net = k_split_augment_with_comparison(
                    network, users, llms, k)
                network.reset_network(llms)

                # 为该k值单独绘图
                plot_withdraw_cost_single_k(distribution_name, k,
                                            backtrack_iters, cost_comparisons)


if __name__ == "__main__":
    main()
