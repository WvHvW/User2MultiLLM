"""
对比 task_offloading_route 与 bottleneck-augment 在同一网络配置下的
总成本、服务率和运行时间，并画图输出。

数据从 sheets 中读取；默认使用 gaussian-gaussian 分布。
"""

import time
import os
from typing import Dict, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import Entity  # noqa: E402
from analyze_bandwidth_llm_combined import run_augment_with_compute_flow_mapping  # noqa: E402


def build_user_ideal_llms(
        network: Entity.Network, users: Dict[int, Entity.User],
        llms: Dict[int, Entity.LLM]) -> Dict[int, Dict[int, float]]:
    """
    基于 dijkstra_ideal，为每个用户构建按距离从近到远排序的理想 LLM 列表。

    返回:
        {user_id: {llm_id: cost, ...}, ...}
    """
    user_ideal_llms: Dict[int, Dict[int, float]] = {}

    for uid, user in users.items():
        distances, costs = network.dijkstra_ideal(uid, user.bw)

        # 仅保留 LLM 节点，按距离从近到远排序
        sorted_llm_ids = sorted(
            llms.keys(),
            key=lambda lid: distances.get(lid, float('inf')),
        )

        ideal_llms: Dict[int, float] = {}
        for lid in sorted_llm_ids:
            if distances.get(lid, float('inf')) == float('inf'):
                continue
            ideal_llms[lid] = costs[lid]

        if ideal_llms:
            user_ideal_llms[uid] = ideal_llms

    return user_ideal_llms


def run_single_case(
        user_distribution: str = 'gaussian',
        llm_distribution: str = 'gaussian',
        flow_per_compute: float = 125.0
) -> Dict[str, Tuple[float, float, float]]:
    """
    在给定的用户 / LLM 分布下，分别运行 task_offloading_route 与 bottleneck-augment，
    返回两者的 (cost, service_rate, runtime)。
    """
    json_data = Entity.load_network_from_sheets()
    network: Entity.Network = json_data['network']

    llms = Entity.load_llm_info(user_distribution, llm_distribution)
    users = Entity.load_user_info(user_distribution)

    # 为每个用户构建理想 LLM 列表
    user_ideal_llms = build_user_ideal_llms(network, users, llms)

    # 运行 task_offloading_route
    t0 = time.perf_counter()
    to_allocs, _, to_cost, to_sr = Entity.task_offloading_route(
        network,
        users,
        llms,
        user_ideal_llms,
        algorithm_name="task-offloading")
    t1 = time.perf_counter()
    to_time = t1 - t0

    # 运行 bottleneck-augment（作为对照）
    t0 = time.perf_counter()
    ba_allocs, _, ba_cost, ba_sr = run_augment_with_compute_flow_mapping(
        network,
        users,
        llms,
        'bottleneck-augment',
        k=1,
        flow_per_compute=flow_per_compute,
    )
    t1 = time.perf_counter()
    ba_time = t1 - t0

    print(f"[{user_distribution}-{llm_distribution}]")
    print(
        f"  task-offloading  : cost={to_cost:.3f}, sr={to_sr:.3f}, time={to_time:.4f}s, "
        f"allocs={len(to_allocs)}")
    print(
        f"  bottleneck-augment: cost={ba_cost:.3f}, sr={ba_sr:.3f}, time={ba_time:.4f}s, "
        f"allocs={len(ba_allocs)}")

    return {
        'task-offloading': (to_cost, to_sr, to_time),
        'bottleneck-augment': (ba_cost, ba_sr, ba_time),
    }


def plot_comparison(result: Dict[str, Tuple[float, float, float]], title: str,
                    filename: str) -> None:
    """
    针对单个分布组合画出成本、服务率、运行时间的对比柱状图。
    """
    algorithms = ['task-offloading', 'bottleneck-augment']
    costs = [result[alg][0] for alg in algorithms]
    srs = [result[alg][1] for alg in algorithms]
    times = [result[alg][2] for alg in algorithms]

    x = np.arange(len(algorithms))
    width = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 成本对比
    axes[0].bar(x, costs, width, color=['tab:blue', 'tab:orange'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algorithms, rotation=20)
    axes[0].set_title('Total Cost')
    axes[0].set_ylabel('Cost')
    axes[0].grid(True, axis='y', alpha=0.3)

    # 服务率对比
    axes[1].bar(x, [sr * 100 for sr in srs],
                width,
                color=['tab:blue', 'tab:orange'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms, rotation=20)
    axes[1].set_title('Service Rate')
    axes[1].set_ylabel('Service Rate (%)')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, axis='y', alpha=0.3)

    # 运行时间对比
    axes[2].bar(x, times, width, color=['tab:blue', 'tab:orange'])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(algorithms, rotation=20)
    axes[2].set_title('Runtime')
    axes[2].set_ylabel('Time (s)')
    axes[2].grid(True, axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图：{filename}")


def main():
    user_distribution = 'poisson'
    llm_distribution = 'uniform'
    flow_per_compute = 125.0

    result = run_single_case(user_distribution=user_distribution,
                             llm_distribution=llm_distribution,
                             flow_per_compute=flow_per_compute)

    title = f"task-offloading vs bottleneck-augment ({user_distribution}-{llm_distribution})"
    filename = os.path.join(
        'visualization',
        f'taskoffload_vs_bottleneck_{user_distribution}-{llm_distribution}.png'
    )
    plot_comparison(result, title, filename)


if __name__ == "__main__":
    main()
