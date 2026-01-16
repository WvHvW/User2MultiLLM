"""
网络规模影响实验

研究目标：
- 分析网络规模变化对cost和acceptance ratio的影响
- 固定用户数(16)和LLM数(4)，只改变网络节点总数

实验设置：
- 网络规模：20, 40, 60, 80, 100, 200, 300, 400节点
- 用户数：16（固定）
- LLM数：4（固定）
- 用户需求：[30, 40, 50] Gbps（随机生成）
- LLM容量：[100, 150, 200] Gbps（随机生成）
- 分布：uniform（固定）
- 网络带宽：固定为100 Gbps
- LLM服务容量：固定为150 Gbps

输出：
- results/scale-impact-results.xlsx
- 字段：网络规模、算法名、开销、运行时间、服务接受率
"""

import os
import time
import copy
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import Entity
import Algorithm

# ========== 实验配置 ==========
NETWORK_SIZES = [20, 40, 60, 80, 100, 200, 300, 400]
FIXED_BANDWIDTH = 100  # 固定网络带宽（Gbps）
FIXED_LLM_CAPACITY = 150  # 固定LLM服务容量（Gbps）

# 所有分布类型（两两组合）
DISTRIBUTION_TYPES = ['uniform', 'poisson', 'sparse', 'gaussian']

# 算法列表
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
        'name': 'bottleneck-split-augment',
        'need_ideal': False
    },
]


def load_network_data_fixed(network_size: int,
                            user_dist: str,
                            llm_dist: str,
                            sheets_root='sheets'):
    """
    加载固定配置的网络数据
    
    Args:
        network_size: 网络节点数
        user_dist: 用户分布类型
        llm_dist: LLM分布类型
        sheets_root: sheets根目录
    
    Returns:
        (network, users, llms, user_ideal_llms)
    """
    network_dir = os.path.join(sheets_root, f'N-{network_size}')

    # 加载用户和LLM信息
    users = Entity.load_user_info(user_dist, sheets_root=network_dir)
    llms = Entity.load_llm_info(user_dist, llm_dist, sheets_root=network_dir)

    # 加载网络拓扑
    json_obj = Entity.load_network_from_sheets(llm_ids=llms.keys(),
                                               sheets_root=network_dir)
    network = json_obj['network']

    # 设置固定的网络带宽
    for src, links in network.links.items():
        for link in links:
            if not link.is_reverse:
                link.capacity = FIXED_BANDWIDTH
                link.flow = 0

    # 设置固定的LLM服务容量
    for llm in llms.values():
        llm.service_capacity = FIXED_LLM_CAPACITY

    # 预计算user_ideal_llms
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
    """
    net_copy = copy.deepcopy(network)
    start_time = time.perf_counter()

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

    end_time = time.perf_counter()
    runtime = end_time - start_time

    return {
        'algorithm': algorithm_name,
        'runtime': runtime,
        'total_cost': result['total_cost'],
        'acceptance_ratio': result['acceptance_ratio'],
    }


def run_scale_experiments(sheets_root='sheets', results_root='results'):
    """
    运行网络规模影响实验（参考Experiment.py架构）
    遍历所有分布组合，结果按"user分布-llm分布"分sheet存储
    """
    print("=" * 80)
    print("网络规模影响实验（所有分布组合）")
    print("=" * 80)
    print(f"实验配置：")
    print(f"  - 网络规模: {NETWORK_SIZES}")
    print(f"  - 用户数: 16 (固定)")
    print(f"  - LLM数: 4 (固定)")
    print(f"  - 网络带宽: {FIXED_BANDWIDTH} Gbps (固定)")
    print(f"  - LLM服务容量: {FIXED_LLM_CAPACITY} Gbps (固定)")
    print(
        f"  - 分布组合: {len(DISTRIBUTION_TYPES)}×{len(DISTRIBUTION_TYPES)} = {len(DISTRIBUTION_TYPES)**2}种"
    )
    print(f"  - 对比算法: {len(ALGORITHMS)}种")
    print("=" * 80)

    # 创建结果目录
    os.makedirs(results_root, exist_ok=True)

    # 输出文件路径
    output_path = os.path.join(results_root, 'scale-impact-results.xlsx')

    # 删除旧文件（如果存在）
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"删除旧文件: {output_path}\n")

    # 统计信息
    total_combinations = len(NETWORK_SIZES) * len(DISTRIBUTION_TYPES) * len(
        DISTRIBUTION_TYPES)
    total_experiments = total_combinations * len(ALGORITHMS)
    current_combination = 0
    current_experiment = 0

    # 遍历每个网络规模和分布组合
    for network_size in NETWORK_SIZES:
        for user_dist in DISTRIBUTION_TYPES:
            for llm_dist in DISTRIBUTION_TYPES:
                current_combination += 1
                print(f"\n{'='*80}")
                print(f"[{current_combination}/{total_combinations}] "
                      f"N-{network_size}, user={user_dist}, llm={llm_dist}")
                print(f"{'='*80}")

                # 加载网络数据
                try:
                    network, users, llms, user_ideal_llms = load_network_data_fixed(
                        network_size, user_dist, llm_dist, sheets_root)
                except Exception as e:
                    print(f"[错误] 加载网络数据失败: {e}")
                    continue

                # 打印网络信息
                total_demand = sum(u.bw for u in users.values())
                total_capacity = sum(l.service_capacity for l in llms.values())
                print(f"  - 用户数: {len(users)}, 总需求: {total_demand} Gbps")
                print(f"  - LLM数: {len(llms)}, 总容量: {total_capacity} Gbps")
                print(f"  - 容量/需求比: {total_capacity / total_demand:.4f}")

                # 当前组合的结果（增量写入）
                combination_results = []

                # 运行所有算法
                for algo_config in ALGORITHMS:
                    current_experiment += 1
                    algo_name = algo_config['name']
                    need_ideal = algo_config['need_ideal']

                    print(
                        f"  [{current_experiment}/{total_experiments}] {algo_name}",
                        end=' ')

                    # 运行实验
                    exp_result = run_single_experiment(network, users, llms,
                                                       user_ideal_llms,
                                                       algo_name, need_ideal)

                    if exp_result is None:
                        print("失败")
                        continue

                    # 记录结果（只保存关键字段）
                    result_record = {
                        '网络规模': network_size,
                        '算法名': algo_name,
                        '开销': exp_result['total_cost'],
                        '运行时间': exp_result['runtime'],
                        '服务接受率': exp_result['acceptance_ratio'],
                    }
                    combination_results.append(result_record)

                    print(f"→ 开销={exp_result['total_cost']:.2f}, "
                          f"服务率={exp_result['acceptance_ratio']:.4f}, "
                          f"时间={exp_result['runtime']:.4f}s")

                # 立即写入当前组合的结果（按user-llm分布分sheet）
                if combination_results:
                    df_new = pd.DataFrame(combination_results)
                    sheet_name = f"{user_dist}-{llm_dist}"[:
                                                           31]  # Excel sheet名限制31字符

                    if os.path.exists(output_path):
                        # 追加模式：读取现有数据并合并
                        with pd.ExcelWriter(
                                output_path,
                                engine='openpyxl',
                                mode='a',
                                if_sheet_exists='overlay') as writer:
                            if sheet_name in writer.book.sheetnames:
                                # 读取现有sheet并合并
                                df_existing = pd.read_excel(
                                    output_path,
                                    sheet_name=sheet_name,
                                    engine='openpyxl')
                                df_combined = pd.concat([df_existing, df_new],
                                                        ignore_index=True)
                                # 删除旧sheet
                                del writer.book[sheet_name]
                                df_combined.to_excel(writer,
                                                     sheet_name=sheet_name,
                                                     index=False)
                            else:
                                df_new.to_excel(writer,
                                                sheet_name=sheet_name,
                                                index=False)
                    else:
                        # 首次写入
                        with pd.ExcelWriter(output_path,
                                            engine='openpyxl') as writer:
                            df_new.to_excel(writer,
                                            sheet_name=sheet_name,
                                            index=False)

                    print(
                        f"  ✓ 结果已保存到sheet: {sheet_name} ({len(combination_results)}条记录)"
                    )

    # 最终汇总
    print(f"\n{'='*80}")
    print(f"所有实验完成！")
    print(f"{'='*80}")

    if os.path.exists(output_path):
        print(f"输出文件: {output_path}")

        # 读取所有sheet并打印汇总
        excel_file = pd.ExcelFile(output_path, engine='openpyxl')
        total_records = 0
        print(f"\nSheet汇总（共{len(excel_file.sheet_names)}个分布组合）:")
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(output_path,
                               sheet_name=sheet_name,
                               engine='openpyxl')
            total_records += len(df)
            print(f"  - {sheet_name}: {len(df)}条记录")
        print(f"\n总实验次数: {total_records}")
    else:
        print("\n[错误] 没有收集到任何实验结果")


if __name__ == '__main__':
    run_scale_experiments()
