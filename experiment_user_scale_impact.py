"""
用户规模影响实验 - 实验运行脚本

研究目标：
- 分析用户数量变化对cost和acceptance ratio的影响
- 固定网络规模(100节点)和LLM数(16)，只改变用户数量

实验设置：
- 网络规模：100节点（固定）
- 用户数：[4, 8, 16, 32, 64]
- LLM数：16（固定）
- 用户需求：50 Gbps（固定）
- LLM容量：100 Gbps（固定）
- 分布：uniform/poisson/sparse/gaussian 两两搭配（16种组合）
- 网络带宽：100 Gbps

输出：
- results/user-scale-impact-results.xlsx
- 按"user分布-llm分布"分sheet存储结果
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
NETWORK_SIZE = 100  # 固定网络规模
USER_COUNTS = [4, 8, 16, 32, 64]  # 用户数量变化
FIXED_NUM_LLMS = 16  # 固定LLM数量
FIXED_USER_DEMAND = 50  # 固定用户需求（Gbps）
FIXED_LLM_CAPACITY = 100  # 固定LLM容量（Gbps）
FIXED_BANDWIDTH = 100  # 固定网络带宽（Gbps）

# 分布类型（两两搭配）
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


def load_network_data(num_users: int,
                      user_dist: str,
                      llm_dist: str,
                      sheets_root='sheets'):
    """
    加载用户规模实验的网络数据
    
    数据结构：
    - sheets/N-100-user-{num_users}/
      - adjacency.xlsx, node.xlsx, bandwidth.xlsx, distance.xlsx（网络拓扑）
      - distribution/{user_dist}.xlsx（用户和LLM分布数据）
        - user sheet: user信息（不同用户数）
        - llm-{llm_dist} sheet: llm信息（固定16个）
    
    Args:
        num_users: 用户数量
        user_dist: 用户分布类型
        llm_dist: LLM分布类型
        sheets_root: sheets根目录
    
    Returns:
        (network, users, llms, user_ideal_llms)
    """
    network_dir = os.path.join(sheets_root,
                               f'N-{NETWORK_SIZE}-user-{num_users}')
    dist_file = os.path.join(network_dir, 'distribution', f'{user_dist}.xlsx')

    # 1. 读取用户信息（从user sheet）
    df_user = pd.read_excel(dist_file, sheet_name='user', engine='openpyxl')
    users = {}
    for _, row in df_user.iterrows():
        node_id = int(row['node_id'])
        users[node_id] = Entity.User(id=node_id, bw=float(row['bw_demand']))

    # 2. 读取LLM信息（从llm-{llm_dist} sheet）
    llm_sheet_name = f'llm-{llm_dist}'
    df_llm = pd.read_excel(dist_file,
                           sheet_name=llm_sheet_name,
                           engine='openpyxl')
    llms = {}
    for _, row in df_llm.iterrows():
        node_id = int(row['node_id'])
        llms[node_id] = Entity.LLM(id=node_id,
                                   service_capacity=float(
                                       row['service_capacity']))

    # 3. 加载网络拓扑
    json_obj = Entity.load_network_from_sheets(llm_ids=llms.keys(),
                                               sheets_root=network_dir)
    network = json_obj['network']

    # 4. 验证固定参数
    for src, links in network.links.items():
        for link in links:
            if not link.is_reverse:
                link.flow = 0  # 重置流量

    # 5. 预计算user_ideal_llms
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
        print(f"失败: {e}")
        return None

    end_time = time.perf_counter()
    runtime = end_time - start_time

    return {
        'algorithm': algorithm_name,
        'runtime': runtime,
        'total_cost': result['total_cost'],
        'acceptance_ratio': result['acceptance_ratio'],
    }


def run_user_scale_experiments(sheets_root='sheets', results_root='results'):
    """
    运行用户规模影响实验
    遍历所有用户数量和分布组合，结果按"user分布-llm分布"分sheet存储
    """
    print("=" * 80)
    print("用户规模影响实验（所有分布组合）")
    print("=" * 80)
    print(f"实验配置：")
    print(f"  - 网络规模: {NETWORK_SIZE}节点（固定）")
    print(f"  - 用户数: {USER_COUNTS}")
    print(f"  - LLM数: {FIXED_NUM_LLMS}（固定）")
    print(f"  - 用户需求: {FIXED_USER_DEMAND} Gbps（固定）")
    print(f"  - LLM容量: {FIXED_LLM_CAPACITY} Gbps（固定）")
    print(f"  - 网络带宽: {FIXED_BANDWIDTH} Gbps（固定）")
    print(
        f"  - 分布组合: {len(DISTRIBUTION_TYPES)}×{len(DISTRIBUTION_TYPES)} = {len(DISTRIBUTION_TYPES)**2}种"
    )
    print(f"  - 对比算法: {len(ALGORITHMS)}种")
    print("=" * 80)

    # 创建结果目录
    os.makedirs(results_root, exist_ok=True)

    # 输出文件路径
    output_path = os.path.join(results_root, 'user-scale-impact-results.xlsx')

    # 删除旧文件（如果存在）
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"删除旧文件: {output_path}\n")

    # 统计信息
    total_combinations = len(USER_COUNTS) * len(DISTRIBUTION_TYPES) * len(
        DISTRIBUTION_TYPES)
    total_experiments = total_combinations * len(ALGORITHMS)
    current_combination = 0
    current_experiment = 0

    # 遍历每个用户数量和分布组合
    for num_users in USER_COUNTS:
        for user_dist in DISTRIBUTION_TYPES:
            for llm_dist in DISTRIBUTION_TYPES:
                current_combination += 1
                print(f"\n{'='*80}")
                print(f"[{current_combination}/{total_combinations}] "
                      f"用户数={num_users}, user={user_dist}, llm={llm_dist}")
                print(f"{'='*80}")

                # 加载网络数据
                try:
                    network, users, llms, user_ideal_llms = load_network_data(
                        num_users, user_dist, llm_dist, sheets_root)
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
                        continue

                    # 记录结果
                    result_record = {
                        '用户数': num_users,
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
    run_user_scale_experiments()
