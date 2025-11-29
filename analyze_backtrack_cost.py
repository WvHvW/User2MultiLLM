"""
分析增广路径算法中回撤流量带来的成本优化。

对比同粒度下 k-split（贪心）vs k-split-augment（增广）的单次推流成本差异，
只统计在增广路径中实际使用了反向边的轮次。

当前版本仅从 sheets 读取 user / LLM 的空间分布（节点位置与个数），
具体的用户带宽 / 计算需求、网络链路带宽以及 LLM 计算容量
均在本脚本中按 (pattern, bandwidth, llm_computation) 组合自行设置，
输出按 pattern / 分布 组织到 withdraw 目录下。
"""

from collections import deque
import copy
import math
import os

import Entity
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# 基础配置
DISTRIBUTION_TYPES = Entity.DISTRIBUTION_TYPES
is_shared = 1
K_VALUES = [1]

# 带宽与 LLM 容量扫描范围（与综合带宽-LLM 分析脚本保持一致）
BANDWIDTH_VALUES = list(range(500, 2000, 500))  # 500-2000，步长 500
COMPUTATION_VALUES = list(range(8, 16, 4))  # 8-16，步长 4

# 用户计算需求 pattern 与带宽映射
USER_COMPUTATION_PATTERNS = [
    [4, 4, 4, 4, 4, 4, 4, 4],
    [6, 4, 4, 4, 4, 4, 4, 2],
    [6, 6, 4, 4, 4, 4, 2, 2],
    [6, 6, 6, 4, 4, 2, 2, 2],
    [6, 6, 6, 6, 2, 2, 2, 2],
]
USER_COMPUTATION_TO_BANDWIDTH = {
    6: 750.0,
    4: 500.0,
    2: 250.0,
}

# 输出目录（根目录）
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'withdraw')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def set_uniform_bandwidth(network, bandwidth_value):
    """将网络中所有正向链路的带宽设置为统一值。"""
    bandwidth_value = float(bandwidth_value)
    for links in network.links.values():
        for link in links:
            if not link.is_reverse:
                link.capacity = bandwidth_value


def set_uniform_computation(llms, computation_value):
    """将所有 LLM 的计算资源设置为统一值。"""
    computation_value = float(computation_value)
    for llm in llms.values():
        llm.computation = computation_value
        llm.available_computation = computation_value


def apply_user_pattern(users, pattern):
    """
    按 pattern 为用户设置计算需求与带宽，仅使用 sheets 提供的空间分布。

    - 先根据原始带宽从大到小排序，保证与其他脚本的用户顺序一致；
    - 重置所有用户的 bw / computation 为 0，只保留本脚本中的配置；
    - 将 pattern 中的计算需求映射为带宽并写回前 len(pattern) 个用户。

    返回：按原始带宽降序排序后的用户字典。
    """
    sorted_items = sorted(users.items(),
                          key=lambda item: item[1].bw,
                          reverse=True)

    if len(sorted_items) < len(pattern):
        raise ValueError(
            f"用户数量 {len(sorted_items)} 小于 pattern 长度 {len(pattern)}")

    # 重置所有用户需求，确保只使用当前脚本的配置
    for user in users.values():
        user.computation = 0.0
        user.bw = 0.0

    for idx, demand in enumerate(pattern):
        uid, user = sorted_items[idx]
        bw_value = USER_COMPUTATION_TO_BANDWIDTH.get(demand)
        if bw_value is None:
            raise ValueError(
                f"找不到 computation={demand} 对应的带宽映射，请检查 USER_COMPUTATION_TO_BANDWIDTH"
            )
        user.computation = float(demand)
        user.bw = float(bw_value)

    ordered_users = {uid: users[uid] for uid, _ in sorted_items}
    return ordered_users


def k_split_with_trace(network, users, llms, k):
    """
    贪心 k-split 算法，记录每次推流后的累计成本轨迹。

    返回:
        allocations: 分配结果列表
        cost_trace: 每次推流后的累计成本列表
        net: 带有流量的网络副本
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
            max_customers = int(llm.computation //
                                single_cpu) if single_cpu > 0 else 0
            net.add_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid in llms.keys():
            net.add_link(lid, T, total_bw, 0)

    allocations = []
    cost_trace = []
    cumulative_cost = 0.0
    remaining = total_bw

    while remaining > 1e-9:
        push = min(k, remaining)

        # Dijkstra 寻找最短路径（只考虑正向边）
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
            "user_id": user_id,
            "llm_id": llm_id,
            "path": node_path[1:-1],
            "cost": path_cost,
            "flow": push
        })
        remaining -= push

    return allocations, cost_trace, net


def k_split_augment_with_comparison(network, users, llms, k):
    """
    增广路径算法，在每次使用反向边时，对比贪心算法的路径成本。

    返回:
        backtrack_iters: 使用反向边的迭代序号列表
        cost_comparisons: 每次使用反向边时的成本对比 (augment_cost, greedy_cost, diff)
        net: 带有流量的网络副本
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
            max_customers = int(llm.computation //
                                single_cpu) if single_cpu > 0 else 0
            net.add_link(lid, T, max_customers * single_bw, 0)
    else:
        for lid in llms.keys():
            net.add_link(lid, T, total_bw, 0)

    INF = math.inf
    EPSILON = 1e-9

    backtrack_iters = []
    cost_comparisons = []
    remaining = total_bw
    iteration = 0

    while remaining > 1e-9:
        iteration += 1
        push = min(k, remaining)

        # 运行 SPFA 寻找最短路径（允许使用反向边）
        dist = {nid: INF for nid in net.nodes}
        prev = {nid: None for nid in net.nodes}
        in_queue = {nid: False for nid in net.nodes}

        dist[S] = 0.0
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
            prev_entry = prev[cur]
            if prev_entry is None:
                break
            prev_node, prev_link = prev_entry
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
            greedy_dist, _ = net.dijkstra_with_capacity(S, push)

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
        if is_shared and node_path:
            user_id = node_path[1]
            llm_id = node_path[-2]
            llms[llm_id].available_computation -= single_cpu

        remaining -= push

    return backtrack_iters, cost_comparisons, net


def plot_withdraw_cost_single_k(distribution_name, k, backtrack_iters,
                                cost_comparisons, output_dir, bandwidth,
                                llm_computation):
    """
    为单个 k 值绘制回撤成本优化图。

    参数:
        distribution_name: 分布名称
        k: 粒度值
        backtrack_iters: 使用反向边的迭代序号列表
        cost_comparisons: 成本对比列表 (augment_cost, greedy_cost, diff)
        output_dir: 输出目录（pattern / distribution 级别）
        bandwidth: 当前统一链路带宽
        llm_computation: 当前统一 LLM 计算容量
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

    # 绘制无解的点（红色 X 标记在 y=0 位置）
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
    ax.set_ylabel('optimization (greedy_cost - augment_cost)', fontsize=13)
    ax.set_title(
        f'{distribution_name} (bw={bandwidth}, llm={llm_computation}, k={k})',
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

    # 保存图片：文件名编码带宽和 LLM 容量设置
    filename = os.path.join(
        output_dir,
        f'bw{int(bandwidth)}_llm{int(llm_computation)}_k{k}.png',
    )
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    已保存 {filename}")


def main():
    """
    pattern × 分布 × (带宽, LLM 容量) × k 四维组合的回撤成本分析：
    - 仅从 sheets 读取 user / LLM 分布；
    - 在本脚本中为每个组合设置统一的链路带宽与 LLM 容量；
    - 用户需求通过 USER_COMPUTATION_PATTERNS 映射；
    - 结果图保存在 withdraw/pattern{idx}/{distribution}/bw{B}_llm{C}_k{k}.png。
    """
    for pattern_index, pattern in enumerate(USER_COMPUTATION_PATTERNS):
        print(f"\n处理 Pattern {pattern_index + 1}: {pattern}")

        for user_distribution in DISTRIBUTION_TYPES:
            for llm_distribution in DISTRIBUTION_TYPES:
                distribution_name = f'{user_distribution}-{llm_distribution}'
                print(f"  处理分布: {distribution_name}")

                pattern_dir = os.path.join(_OUTPUT_DIR,
                                           f'pattern{pattern_index + 1}',
                                           distribution_name)
                os.makedirs(pattern_dir, exist_ok=True)

                for bandwidth in BANDWIDTH_VALUES:
                    for computation in COMPUTATION_VALUES:
                        print(
                            f"    带宽={bandwidth}, LLM容量={computation} 下分析回撤成本..."
                        )

                        # 仅从 sheets 读取拓扑与 user/LLM 分布
                        json_data = Entity.load_network_from_sheets()
                        network = json_data['network']
                        llms = Entity.load_llm_info(user_distribution,
                                                    llm_distribution)
                        users = Entity.load_user_info(user_distribution)

                        # 应用 pattern 与统一带宽 / 计算资源配置
                        users = apply_user_pattern(users, pattern)
                        set_uniform_bandwidth(network, bandwidth)
                        set_uniform_computation(llms, computation)

                        # 在相同 (pattern, 分布, 带宽, 计算) 下考察不同 k
                        for k in K_VALUES:
                            backtrack_iters, cost_comparisons, _ = k_split_augment_with_comparison(
                                network, users, llms, k)

                            plot_withdraw_cost_single_k(
                                distribution_name=distribution_name,
                                k=k,
                                backtrack_iters=backtrack_iters,
                                cost_comparisons=cost_comparisons,
                                output_dir=pattern_dir,
                                bandwidth=bandwidth,
                                llm_computation=computation,
                            )

                            # 重置网络与 LLM 可用资源，保证同一组合下不同 k 之间互不干扰
                            network.reset_network(llms)


if __name__ == "__main__":
    main()
