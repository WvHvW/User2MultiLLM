"""
用户规模影响实验 - 数据生成脚本

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
- sheets/N-100-user-{num_users}/distribution/{user_dist}.xlsx
  - user sheet: num_users个用户
  - llm-{llm_dist} sheet: 16个LLM（4种分布）
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Tuple
import PN

# ========== 实验配置 ==========
NETWORK_SIZE = 100  # 固定网络规模
USER_COUNTS = [4, 8, 16, 32, 64]  # 用户数量变化
FIXED_NUM_LLMS = 8  # 固定LLM数量
FIXED_USER_DEMAND = 50  # 固定用户需求（Gbps）
FIXED_LLM_CAPACITY = 100  # 固定LLM容量（Gbps）
FIXED_BANDWIDTH = 100  # 固定网络带宽（Gbps）

# 分布类型（两两搭配）
DISTRIBUTION_TYPES = ['uniform', 'poisson', 'sparse', 'gaussian']

# 网络拓扑参数（参考PN.py）
TARGET_DEGREE = [3, 4, 5]  # 度数约束列表


def save_network_to_sheets(G: nx.Graph, sheets_root: str):
    """
    将网络图保存为sheets格式（与generate_fixed_user_llm.py一致）
    
    Args:
        G: NetworkX图
        sheets_root: sheets根目录
    """
    os.makedirs(sheets_root, exist_ok=True)

    # 输出邻接矩阵
    adjacency_matrix = pd.DataFrame(np.zeros((len(G.nodes()), len(G.nodes()))),
                                    index=list(G.nodes()),
                                    columns=list(G.nodes()))
    for u, v, attr in G.edges(data=True):
        adjacency_matrix.at[u, v] = 1
        adjacency_matrix.at[v, u] = 1
    adjacency_matrix.to_excel(os.path.join(sheets_root, 'adjacency.xlsx'))

    # 输出节点数据（包含坐标）
    node_data = []
    for n, attr in G.nodes(data=True):
        node_info = {'node_id': n}
        node_info.update(attr)
        if 'pos' in node_info:
            node_info['pos_x'] = node_info['pos'][0]
            node_info['pos_y'] = node_info['pos'][1]
            del node_info['pos']
        node_data.append(node_info)
    node_df = pd.DataFrame(node_data)
    node_df.to_excel(os.path.join(sheets_root, 'node.xlsx'), index=False)

    # 输出带宽矩阵
    bandwidth_matrix = pd.DataFrame(np.zeros((len(G.nodes()), len(G.nodes()))),
                                    index=list(G.nodes()),
                                    columns=list(G.nodes()))
    for u, v, attr in G.edges(data=True):
        bandwidth_matrix.at[u, v] = attr['capacity_mbps']
        bandwidth_matrix.at[v, u] = attr['capacity_mbps']
    bandwidth_matrix.to_excel(os.path.join(sheets_root, 'bandwidth.xlsx'))

    # 输出距离矩阵
    distance_matrix = pd.DataFrame(np.zeros((len(G.nodes()), len(G.nodes()))),
                                   index=list(G.nodes()),
                                   columns=list(G.nodes()))
    for u, v, attr in G.edges(data=True):
        distance_matrix.at[u, v] = attr['distance']
        distance_matrix.at[v, u] = attr['distance']
    distance_matrix.to_excel(os.path.join(sheets_root, 'distance.xlsx'))


def assign_nodes_by_distribution(G: nx.Graph,
                                 num_nodes: int,
                                 distribution: str,
                                 exclude_nodes: set = None) -> List[int]:
    """
    根据分布类型分配节点
    
    Args:
        G: 网络图
        num_nodes: 需要分配的节点数
        distribution: 分布类型（uniform/poisson/sparse/gaussian）
        exclude_nodes: 需要排除的节点集合
    
    Returns:
        选中的节点列表
    """
    if exclude_nodes is None:
        exclude_nodes = set()

    available_nodes = [n for n in G.nodes() if n not in exclude_nodes]
    n = len(available_nodes)

    if num_nodes > len(available_nodes):
        raise ValueError(f"需要{num_nodes}个节点，但只有{len(available_nodes)}个可用节点")

    if distribution == 'uniform':
        # 均匀分布：每个节点权重相同
        weights = np.ones(n)

    elif distribution == 'poisson':
        # 泊松分布：中心节点权重高
        center_idx = n // 2
        distances = np.abs(np.arange(n) - center_idx)
        weights = np.exp(-distances / (n / 4))

    elif distribution == 'sparse':
        # 稀疏分布：边缘节点权重高
        center_idx = n // 2
        distances = np.abs(np.arange(n) - center_idx)
        weights = distances + 1

    elif distribution == 'gaussian':
        # 高斯分布：中心附近节点权重高
        center_idx = n // 2
        distances = np.abs(np.arange(n) - center_idx)
        weights = np.exp(-(distances**2) / (2 * (n / 6)**2))

    else:
        raise ValueError(f"未知的分布类型: {distribution}")

    # 归一化权重
    weights = weights / weights.sum()

    # 按权重采样
    selected_indices = np.random.choice(n,
                                        size=num_nodes,
                                        replace=False,
                                        p=weights)
    selected_nodes = [available_nodes[i] for i in selected_indices]

    return selected_nodes


def generate_data_for_user_count(num_users: int,
                                 user_dist: str,
                                 llm_dist: str,
                                 G: nx.Graph,
                                 sheets_root='sheets'):
    """
    为特定用户数量和分布组合生成数据
    
    Args:
        num_users: 用户数量
        user_dist: 用户分布类型
        llm_dist: LLM分布类型
        G: 网络图（复用，避免重复生成）
        sheets_root: sheets根目录
    """
    # 分配用户节点
    user_nodes = assign_nodes_by_distribution(G, num_users, user_dist)

    # 分配LLM节点（排除用户节点）
    llm_nodes = assign_nodes_by_distribution(G,
                                             FIXED_NUM_LLMS,
                                             llm_dist,
                                             exclude_nodes=set(user_nodes))

    # 创建用户数据（与Entity.py格式一致）
    user_data = []
    for i, node_id in enumerate(user_nodes):
        user_data.append({
            'node_id': node_id,
            'bw_demand': FIXED_USER_DEMAND,
        })

    # 创建LLM数据（与Entity.py格式一致）
    llm_data = []
    for i, node_id in enumerate(llm_nodes):
        llm_data.append({
            'node_id': node_id,
            'service_capacity': FIXED_LLM_CAPACITY,
        })

    # 构建输出目录
    network_dir = os.path.join(sheets_root,
                               f'N-{NETWORK_SIZE}-user-{num_users}')
    dist_dir = os.path.join(network_dir, 'distribution')
    os.makedirs(dist_dir, exist_ok=True)

    # 输出文件路径
    output_file = os.path.join(dist_dir, f'{user_dist}.xlsx')

    # 写入Excel（user sheet + llm-{llm_dist} sheet）
    df_user = pd.DataFrame(user_data)
    df_llm = pd.DataFrame(llm_data)

    if os.path.exists(output_file):
        # 文件存在：读取并添加新的llm sheet
        with pd.ExcelWriter(output_file,
                            engine='openpyxl',
                            mode='a',
                            if_sheet_exists='replace') as writer:
            # 始终重写user sheet，确保字段名正确
            df_user.to_excel(writer, sheet_name='user', index=False)
            # 添加或更新llm sheet
            df_llm.to_excel(writer, sheet_name=f'llm-{llm_dist}', index=False)
    else:
        # 文件不存在：创建新文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_user.to_excel(writer, sheet_name='user', index=False)
            df_llm.to_excel(writer, sheet_name=f'llm-{llm_dist}', index=False)


def generate_network_for_user_count(num_users: int, sheets_root='sheets'):
    """
    为特定用户数量生成网络拓扑和所有分布组合的数据
    
    Args:
        num_users: 用户数量
        sheets_root: sheets根目录
    """
    print(f"\n生成 {num_users} 用户的数据...")

    # 生成网络拓扑（复用一个网络，避免重复生成）
    G, _ = PN.generate_city_network(num_nodes=NETWORK_SIZE,
                                    target_degree=TARGET_DEGREE)

    # 保存网络拓扑和边数据
    network_dir = os.path.join(sheets_root,
                               f'N-{NETWORK_SIZE}-user-{num_users}')
    os.makedirs(network_dir, exist_ok=True)

    # 保存网络数据（与generate_fixed_user_llm.py格式一致）
    save_network_to_sheets(G, network_dir)
    print(f"  ✓ 网络拓扑已保存到 {network_dir}")

    # 生成所有分布组合（4×4=16种）
    total_combinations = len(DISTRIBUTION_TYPES) * len(DISTRIBUTION_TYPES)
    current = 0

    for user_dist in DISTRIBUTION_TYPES:
        for llm_dist in DISTRIBUTION_TYPES:
            current += 1
            print(
                f"  [{current}/{total_combinations}] user={user_dist}, llm={llm_dist}"
            )

            try:
                generate_data_for_user_count(num_users, user_dist, llm_dist, G,
                                             sheets_root)
            except Exception as e:
                print(f"    [错误] 生成失败: {e}")
                continue

    print(f"  ✓ 完成 {num_users} 用户的所有分布组合")


def generate_all_data(sheets_root='sheets'):
    """
    生成所有用户数量的数据
    """
    print("=" * 80)
    print("用户规模影响实验 - 数据生成")
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
    print("=" * 80)

    # 容量/需求比分析
    print("\n容量/需求比分析：")
    total_capacity = FIXED_NUM_LLMS * FIXED_LLM_CAPACITY
    print(f"  总LLM容量: {total_capacity} Gbps")
    for num_users in USER_COUNTS:
        total_demand = num_users * FIXED_USER_DEMAND
        ratio = total_capacity / total_demand
        print(f"  {num_users}用户: {total_demand} Gbps → 容量/需求比 = {ratio:.2f}")

    print("\n" + "=" * 80)

    # 生成所有用户数量的数据
    total_user_counts = len(USER_COUNTS)
    for idx, num_users in enumerate(USER_COUNTS, 1):
        print(f"\n[{idx}/{total_user_counts}] 生成用户数量: {num_users}")
        try:
            generate_network_for_user_count(num_users, sheets_root)
        except Exception as e:
            print(f"  [错误] 生成失败: {e}")
            continue

    print("\n" + "=" * 80)
    print("所有数据生成完成！")
    print("=" * 80)

    # 打印输出文件结构
    print("\n生成的文件结构：")
    for num_users in USER_COUNTS:
        network_dir = os.path.join(sheets_root,
                                   f'N-{NETWORK_SIZE}-user-{num_users}')
        if os.path.exists(network_dir):
            print(f"  {network_dir}/")
            print(f"    network.json")
            dist_dir = os.path.join(network_dir, 'distribution')
            if os.path.exists(dist_dir):
                excel_files = [
                    f for f in os.listdir(dist_dir) if f.endswith('.xlsx')
                ]
                for excel_file in excel_files:
                    print(f"    distribution/{excel_file}")


if __name__ == '__main__':
    generate_all_data()
