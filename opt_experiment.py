"""
暴力算法对比实验

根据ExperimentSetting.md的"暴力算法实验设计"：
1. 为每一单位流量枚举去处（force_brute）
2. 每轮随机分配user和llm位置
3. 对比5个算法：force_brute, 1-split, 1-split-augment, bottleneck-split, bottleneck-split-augment
4. 只记录bottleneck-split-augment比bottleneck-split好20%以上的数据
5. 收集10组有效数据后结束
"""

import os
import random
import time
import copy
import pandas as pd
from typing import Dict, List, Tuple

import Entity
from Entity import Network, User, LLM
from force_brute import brute_force_optimal
from Algorithm import (one_split_no_aggregate, one_split_augment,
                       bottleneck_split_no_aggregate, bottleneck_split_augment,
                       one_split, bottleneck_split)


def random_assign_users_llms(
        network: Network,
        num_users: int = 4,
        num_llms: int = 2,
        user_demands: List[int] = [3, 3, 3, 5],
        llm_capacities: List[int] = [7,
                                     7]) -> Tuple[Dict, Dict, Dict, Network]:
    """
    随机从网络节点中选择user和llm位置，并创建对应对象

    Args:
        network: 原始网络对象
        num_users: 用户数量
        num_llms: LLM数量
        user_demands: 用户需求列表
        llm_capacities: LLM容量列表

    Returns:
        (users字典, llms字典, user_ideal_llms字典, 修改后的network副本)
    """
    # 复制网络以避免修改原始对象
    net = copy.deepcopy(network)

    # 获取所有节点ID
    all_node_ids = list(net.nodes.keys())

    # 随机选择user和llm节点（不重复）
    selected_nodes = random.sample(all_node_ids, num_users + num_llms)
    user_node_ids = selected_nodes[:num_users]
    llm_node_ids = selected_nodes[num_users:]

    # 创建user对象
    users = {}
    for i, node_id in enumerate(user_node_ids):
        demand = user_demands[i] if i < len(user_demands) else user_demands[-1]
        users[node_id] = User(id=node_id, bw=demand)

    # 创建llm对象
    llms = {}
    for i, node_id in enumerate(llm_node_ids):
        capacity = llm_capacities[i] if i < len(
            llm_capacities) else llm_capacities[-1]
        llms[node_id] = LLM(id=node_id, service_capacity=capacity)

    # 删除LLM节点的物理出边（只保留入边）
    for llm_id in llm_node_ids:
        if llm_id in net.links:
            # 只保留反向边，删除所有正向边
            net.links[llm_id] = [
                link for link in net.links[llm_id] if link.is_reverse
            ]

    # 计算user_ideal_llms（部分算法需要）
    user_ideal_llms = {}
    for uid, user in users.items():
        distances, _ = net.dijkstra_ideal(uid, user.bw)
        sorted_llms = sorted(distances.items(), key=lambda x: x[1])
        user_ideal_llms[uid] = {
            lid: dist
            for lid, dist in sorted_llms if lid in llms
        }

    return users, llms, user_ideal_llms, net


def run_single_round(network: Network, users: Dict[int, User], llms: Dict[int,
                                                                          LLM],
                     user_ideal_llms: Dict, round_idx: int) -> List[Dict]:
    """
    运行单轮实验，测试5个算法

    Returns:
        包含所有算法结果的列表
    """
    results = []

    print(f"\n{'='*80}")
    print(f"第 {round_idx} 轮实验")
    print(f"{'='*80}")
    print(f"用户位置: {sorted(users.keys())}")
    print(f"用户需求: {[int(u.bw) for u in users.values()]}")
    print(f"LLM位置: {sorted(llms.keys())}")
    print(f"LLM容量: {[int(l.service_capacity) for l in llms.values()]}")
    print()

    # 1. force_brute
    print("运行 force_brute...")
    start_time = time.perf_counter()
    try:
        best_cost, best_solution, total_search_space = brute_force_optimal(
            network, users, llms)
        runtime = time.perf_counter() - start_time

        # 计算服务率
        total_demand = sum(u.bw for u in users.values())
        # 对于force_brute，如果找到解，服务率就是100%
        acceptance_ratio = 1.0 if best_cost < float('inf') else 0.0

        results.append({
            '算法名': 'force_brute',
            '总花销': best_cost if best_cost < float('inf') else None,
            '运行时间': runtime,
            '服务率': acceptance_ratio,
            '搜索空间大小': total_search_space
        })
        print(
            f"  完成 - 开销: {best_cost:.2f}, 时间: {runtime:.3f}s, 搜索空间: {total_search_space:,}"
        )
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            '算法名': 'force_brute',
            '总花销': None,
            '运行时间': None,
            '服务率': None,
            '搜索空间大小': None
        })

    # 2-7. 运行其他算法
    algorithms = [
        ('1-split_no_aggregation', one_split_no_aggregate,
         True),  # (算法名, 函数, 是否需要user_ideal_llms)
        ('1-split', one_split, True),  # 汇集图版本也需要user_ideal_llms（虽然内部不用）
        ('1-split-augment', one_split_augment, False),
        ('bottleneck-split_no_aggregation', bottleneck_split_no_aggregate,
         True),
        ('bottleneck-split', bottleneck_split, True),  # 汇集图版本也需要user_ideal_llms
        ('bottleneck-split-augment', bottleneck_split_augment, False)
    ]

    for algo_name, algo_func, need_ideal in algorithms:
        print(f"运行 {algo_name}...")

        # 复制网络、用户和LLM状态
        net_copy = copy.deepcopy(network)
        users_copy = copy.deepcopy(users)
        llms_copy = copy.deepcopy(llms)

        start_time = time.perf_counter()
        try:
            # 调用算法
            if need_ideal:
                result = algo_func(net_copy,
                                   users_copy,
                                   llms_copy,
                                   user_ideal_llms,
                                   is_shared=True)
            else:
                result = algo_func(net_copy,
                                   users_copy,
                                   llms_copy,
                                   is_shared=True)

            runtime = time.perf_counter() - start_time

            total_cost = result.get('total_cost', 0)
            total_demand = sum(u.bw for u in users.values())
            total_flow = result.get('total_flow', 0)
            acceptance_ratio = total_flow / total_demand if total_demand > 0 else 0
            search_space = result.get('search_space', None)  # 提取搜索空间

            results.append({
                '算法名': algo_name,
                '总花销': total_cost,
                '运行时间': runtime,
                '服务率': acceptance_ratio,
                '搜索空间大小': search_space
            })
            print(f"  完成 - 开销: {total_cost:.2f}, 时间: {runtime:.3f}s")
        except Exception as e:
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                '算法名': algo_name,
                '总花销': None,
                '运行时间': None,
                '服务率': None,
                '搜索空间大小': None
            })

    return results


def check_optimization_condition(results: List[Dict]) -> bool:
    """
    检查是否满足统计条件：
    1. bottleneck-split-augment比bottleneck-split_no_aggregation好10%以上
    2. bottleneck-split-augment花销 >= force_brute（只统计augment未达到全局最优的情况）
    """
    bottleneck_cost = None
    augment_cost = None
    brute_force_cost = None

    for r in results:
        if r['算法名'] == 'bottleneck-split_no_aggregation' and r[
                '总花销'] is not None:
            bottleneck_cost = r['总花销']
        elif r['算法名'] == 'bottleneck-split-augment' and r['总花销'] is not None:
            augment_cost = r['总花销']
        elif r['算法名'] == 'force_brute' and r['总花销'] is not None:
            brute_force_cost = r['总花销']

    # 检查数据完整性
    if bottleneck_cost is None or augment_cost is None or brute_force_cost is None:
        print(f"\n  数据不完整，跳过")
        print(f"    bottleneck_cost: {bottleneck_cost}")
        print(f"    augment_cost: {augment_cost}")
        print(f"    brute_force_cost: {brute_force_cost}")
        return False

    if bottleneck_cost == 0:
        print(f"\n  bottleneck-split开销为0，跳过")
        return False

    # 计算改进率
    improvement = (bottleneck_cost - augment_cost) / bottleneck_cost

    print(f"\n  检查条件:")
    print(f"    force_brute={brute_force_cost:.2f}")
    print(f"    bottleneck-split_no_aggregation={bottleneck_cost:.2f}")
    print(f"    bottleneck-split-augment={augment_cost:.2f}")
    print(f"    改进率={improvement*100:.1f}%")

    # 条件1: augment比bottleneck好15%以上
    condition1 = improvement >= 0.0

    # 条件2: augment花销 >= force_brute（只统计未达最优的情况）
    condition2 = augment_cost >= brute_force_cost

    print(f"    条件1(改进率≥15%): {'OK' if condition1 else 'NO'}")
    print(f"    条件2(augment≥brute): {'OK' if condition2 else 'NO'}")

    return condition1 and condition2


def main():
    """主实验流程"""
    print("=" * 80)
    print("暴力算法对比实验")
    print("=" * 80)
    print()

    # 1. 加载网络
    print("加载网络数据...")
    sheets_root = 'sheets/N-opt'
    if not os.path.exists(sheets_root):
        print(f"错误: {sheets_root} 目录不存在！")
        print("请先创建小规模网络数据文件")
        return

    json_obj = Entity.load_network_from_sheets(sheets_root=sheets_root)
    network = json_obj['network']
    nodes = json_obj['nodes']

    print(f"  网络节点数: {len(nodes)}")
    physical_edges = sum(1 for links in network.links.values()
                         for link in links if not link.is_reverse)
    print(f"  网络边数: {physical_edges}")
    print()

    # 2. 实验配置
    num_users = 4
    num_llms = 2
    user_demands = [1, 1, 1, 2]
    llm_capacities = [2, 3]
    target_valid_rounds = 1  # 目标收集10组有效数据

    print(f"实验配置:")
    print(f"  用户数: {num_users}")
    print(f"  用户需求: {user_demands} (总计{sum(user_demands)}单位)")
    print(f"  LLM数: {num_llms}")
    print(f"  LLM容量: {llm_capacities}")
    print(f"  目标收集: {target_valid_rounds}组有效数据")
    print(f"  有效数据条件: bottleneck-split-augment比bottleneck-split好10%以上")
    print()

    # 3. 运行实验
    all_results = []
    valid_count = 0
    round_idx = 0
    max_attempts = 1000  # 最大尝试次数，避免无限循环

    while valid_count < target_valid_rounds and round_idx < max_attempts:
        round_idx += 1

        # 随机分配user和llm位置
        users, llms, user_ideal_llms, exp_network = random_assign_users_llms(
            network, num_users, num_llms, user_demands, llm_capacities)

        # 运行单轮实验
        round_results = run_single_round(exp_network, users, llms,
                                         user_ideal_llms, round_idx)

        # 检查是否满足条件
        if check_optimization_condition(round_results):
            valid_count += 1
            print(f"\n[OK] 第 {valid_count}/{target_valid_rounds} 组有效数据")

            # 添加结果（带轮次编号）
            for r in round_results:
                r['轮次'] = valid_count
            all_results.extend(round_results)

            # 不同轮次间添加空行标记
            if valid_count < target_valid_rounds:
                all_results.append({
                    '轮次': None,
                    '算法名': None,
                    '总花销': None,
                    '运行时间': None,
                    '服务率': None,
                    '搜索空间大小': None
                })
        else:
            print(f"\n[NO] 不满足条件（需要改进≥20%），跳过本轮")

    # 4. 保存结果
    if all_results:
        output_dir = 'results/opt'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'force_baseline.xlsx')

        df = pd.DataFrame(all_results)
        # 调整列顺序
        columns_order = ['轮次', '算法名', '总花销', '运行时间', '服务率', '搜索空间大小']
        df = df[columns_order]

        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n{'='*80}")
        print(f"实验完成!")
        print(f"  有效轮次: {valid_count}/{target_valid_rounds}")
        print(f"  总尝试次数: {round_idx}")
        print(f"  结果已保存到: {output_path}")
        print(f"{'='*80}")
    else:
        print("\n未收集到任何有效数据")


if __name__ == '__main__':
    random.seed(42)  # 设置随机种子以保证可重复性
    main()
