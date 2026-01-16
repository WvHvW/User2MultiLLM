"""
生成固定16用户+4LLM的实验数据

实验设置：
- 用户数：16（固定）
- LLM数：4（固定）
- 用户需求：从[30, 40, 50]中随机选择
- LLM容量：从[100, 150, 200]中随机选择
- 分布：用户和LLM都采用uniform分布
- 网络规模：20, 40, 60, 80, 100, 200, 300, 400节点

用途：研究网络规模变化对cost和acceptance ratio的影响
"""

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from PN import generate_city_network

# ========== 固定配置 ==========
FIXED_NUM_USERS = 16
FIXED_NUM_LLMS = 4

# 固定用户需求和LLM容量（消除随机性，确保容量/需求比稳定）
FIXED_USER_BW = 40  # Gbps（所有用户固定40Gbps）
FIXED_LLM_CAPACITY = 150  # Gbps（所有LLM固定150Gbps）
# 总需求：16 × 40 = 640 Gbps
# 总容量：4 × 150 = 600 Gbps
# 容量/需求比：0.9375（轻微欠容量，测试算法优化能力）

# 所有分布类型（两两组合）
DISTRIBUTION_TYPES = ['uniform', 'poisson', 'sparse', 'gaussian']

# 网络规模（与主实验保持一致）
NETWORK_SIZES = [20, 40, 60, 80, 100, 200, 300, 400]

# 每种规模对应的度数约束
DEGREE_CONSTRAINTS = {
    20: [3, 4, 5],
    40: [3, 4, 5],
    60: [3, 4, 5],
    80: [3, 4, 5],
    100: [5, 6, 7],
    200: [5, 6, 7],
    300: [5, 6, 7],
    400: [5, 6, 7]
}


def assign_nodes_by_distribution(G,
                                 distribution_type,
                                 num_nodes,
                                 role='user',
                                 exclude_nodes=None):
    """
    根据分布类型分配节点
    
    Args:
        G: NetworkX图对象
        distribution_type: 分布类型 ('uniform', 'sparse', 'gaussian', 'poisson')
        num_nodes: 要分配的节点数量
        role: 角色 ('user' 或 'llm')
        exclude_nodes: 已分配的节点列表（避免重复）
    
    Returns:
        selected_nodes: 分配的节点列表
    """
    all_nodes = list(G.nodes())

    # 排除已分配的节点
    if exclude_nodes:
        available_nodes = [n for n in all_nodes if n not in exclude_nodes]
    else:
        available_nodes = all_nodes.copy()

    # 检查节点数是否足够
    if len(available_nodes) < num_nodes:
        raise ValueError(f"可用节点不足：需要 {num_nodes}，剩余 {len(available_nodes)}")

    # 根据分布类型计算权重
    if distribution_type == 'uniform':
        weights = np.ones(len(available_nodes))
    elif distribution_type == 'sparse':
        # 稀疏分布：外围区域权重高
        center_x, center_y = 50, 50
        weights = []
        for node_id in available_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            weight = np.exp(distance_to_center / 15)
            weights.append(weight)
        weights = np.array(weights)
    elif distribution_type == 'gaussian':
        # 高斯分布：user集中在左下角(25, 25)，llm集中在右上角(75, 75)
        if role == 'user':
            center_x, center_y = 25, 25
        else:  # llm
            center_x, center_y = 75, 75
        weights = []
        for node_id in available_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            weight = np.exp(-0.5 * (distance_to_center / 10)**2)
            weights.append(weight)
        weights = np.array(weights)
    elif distribution_type == 'poisson':
        # 泊松分布
        center_x, center_y = 50, 50
        lambda_param = 5
        weights = []
        for node_id in available_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            k = int(distance_to_center / 3)
            from math import factorial, exp
            if k > 20:
                k = 20
            poisson_prob = (lambda_param**k *
                            exp(-lambda_param)) / factorial(k)
            weights.append(poisson_prob)
        weights = np.array(weights)
    else:
        raise ValueError(f"不支持的分布类型: {distribution_type}")

    # 归一化权重
    weights = weights / np.sum(weights)

    # 采样
    selected_indices = np.random.choice(len(available_nodes),
                                        size=num_nodes,
                                        replace=False,
                                        p=weights)
    selected_nodes = [available_nodes[i] for i in selected_indices]

    return selected_nodes


def generate_data_for_size(network_size: int, user_dist: str, llm_dist: str,
                           Graph):
    """
    为指定网络规模和分布组合生成用户/LLM数据
    
    Args:
        network_size: 网络节点数
        user_dist: 用户分布类型
        llm_dist: LLM分布类型
        Graph: 网络拓扑图对象
    
    Returns:
        (user_info, llm_info): 用户和LLM信息列表
    """
    G = Graph.copy()

    # 分配用户节点
    user_nodes = assign_nodes_by_distribution(G,
                                              user_dist,
                                              FIXED_NUM_USERS,
                                              role='user',
                                              exclude_nodes=None)
    for node_id in user_nodes:
        G.nodes[node_id]['bw_demand'] = FIXED_USER_BW

    # 分配LLM节点（避免与用户节点重叠）
    llm_nodes = assign_nodes_by_distribution(G,
                                             llm_dist,
                                             FIXED_NUM_LLMS,
                                             role='llm',
                                             exclude_nodes=user_nodes)
    for node_id in llm_nodes:
        G.nodes[node_id]['bw_capacity'] = FIXED_LLM_CAPACITY
        G.nodes[node_id]['deployed'] = 0

    # 提取用户信息
    user_info = []
    for node_id in user_nodes:
        info = {'node_id': node_id, 'bw_demand': G.nodes[node_id]['bw_demand']}
        user_info.append(info)

    # 提取LLM信息
    llm_info = []
    for node_id in llm_nodes:
        info = {
            'node_id': node_id,
            'bw_capacity': G.nodes[node_id]['bw_capacity'],
        }
        llm_info.append(info)

    return user_info, llm_info


def generate_network_for_size(network_size: int):
    """
    为指定网络规模生成所有分布组合的数据
    
    Args:
        network_size: 网络节点数
    """
    print(f"\n{'='*60}")
    print(f"生成 N-{network_size} 节点网络数据")
    print(f"{'='*60}")

    # 获取度数约束
    degree_constraints = DEGREE_CONSTRAINTS[network_size]

    # 生成网络拓扑
    print(f"节点数: {network_size}, 度数约束: {degree_constraints}")
    print(f"User数: {FIXED_NUM_USERS}, LLM数: {FIXED_NUM_LLMS}")
    Graph, _ = generate_city_network(num_nodes=network_size,
                                     target_degree=degree_constraints)

    # 创建输出目录
    output_dir = os.path.join('sheets', f'N-{network_size}')
    dist_dir = os.path.join(output_dir, 'distribution')
    os.makedirs(dist_dir, exist_ok=True)

    # 为每种用户分布生成数据
    for user_dist in DISTRIBUTION_TYPES:
        print(f"\n  用户分布: {user_dist}")

        # 存储当前用户分布下的所有LLM分布数据
        user_data = {}
        llm_data = {}

        # 为每种LLM分布生成数据
        for llm_dist in DISTRIBUTION_TYPES:
            print(f"    LLM分布: {llm_dist}")
            user_info, llm_info = generate_data_for_size(
                network_size, user_dist, llm_dist, Graph)

            # 第一次遇到此user_dist时保存user信息
            if not user_data:
                user_data = user_info

            # 保存当前llm_dist的llm信息
            llm_data[llm_dist] = llm_info

        # 保存到Excel（文件名为user分布）
        dist_file_path = os.path.join(dist_dir, f'{user_dist}.xlsx')
        with pd.ExcelWriter(dist_file_path, engine='openpyxl') as writer:
            # 第一个sheet：user信息
            df_user = pd.DataFrame(user_data)
            df_user.to_excel(writer, sheet_name='user', index=False)

            # 后续sheet：各个llm分布的llm信息
            for llm_dist, llm_info in llm_data.items():
                df_llm = pd.DataFrame(llm_info)
                df_llm.to_excel(writer, sheet_name=llm_dist, index=False)

        print(f"    ✓ 保存: {dist_file_path}")

    # 输出邻接矩阵
    adjacency_matrix = pd.DataFrame(np.zeros(
        (len(Graph.nodes()), len(Graph.nodes()))),
                                    index=list(Graph.nodes()),
                                    columns=list(Graph.nodes()))
    for u, v, attr in Graph.edges(data=True):
        adjacency_matrix.at[u, v] = 1
        adjacency_matrix.at[v, u] = 1
    adj_path = os.path.join(output_dir, 'adjacency.xlsx')
    adjacency_matrix.to_excel(adj_path)
    print(f"  邻接矩阵: {adj_path}")

    # 输出节点数据（包含坐标）
    node_data = []
    for n, attr in Graph.nodes(data=True):
        node_info = {'node_id': n}
        node_info.update(attr)
        if 'pos' in node_info:
            node_info['pos_x'] = node_info['pos'][0]
            node_info['pos_y'] = node_info['pos'][1]
            del node_info['pos']
        node_data.append(node_info)
    node_df = pd.DataFrame(node_data)
    node_path = os.path.join(output_dir, 'node.xlsx')
    node_df.to_excel(node_path, index=False)
    print(f"  节点数据: {node_path}")

    # 输出带宽矩阵
    bandwidth_matrix = pd.DataFrame(np.zeros(
        (len(Graph.nodes()), len(Graph.nodes()))),
                                    index=list(Graph.nodes()),
                                    columns=list(Graph.nodes()))
    for u, v, attr in Graph.edges(data=True):
        bandwidth_matrix.at[u, v] = attr['capacity_mbps']
        bandwidth_matrix.at[v, u] = attr['capacity_mbps']
    bw_path = os.path.join(output_dir, 'bandwidth.xlsx')
    bandwidth_matrix.to_excel(bw_path)
    print(f"  带宽矩阵: {bw_path}")

    # 输出距离矩阵
    distance_matrix = pd.DataFrame(np.zeros(
        (len(Graph.nodes()), len(Graph.nodes()))),
                                   index=list(Graph.nodes()),
                                   columns=list(Graph.nodes()))
    for u, v, attr in Graph.edges(data=True):
        distance_matrix.at[u, v] = attr['distance']
        distance_matrix.at[v, u] = attr['distance']
    dist_path = os.path.join(output_dir, 'distance.xlsx')
    distance_matrix.to_excel(dist_path)
    print(f"  距离矩阵: {dist_path}")

    # 打印网络统计信息
    print(f"\n  网络统计:")
    print(f"    - 节点数: {Graph.number_of_nodes()}")
    print(f"    - 边数: {Graph.number_of_edges()}")
    print(
        f"    - 平均度: {2 * Graph.number_of_edges() / Graph.number_of_nodes():.2f}"
    )
    print(f"    - 总需求: {FIXED_NUM_USERS * FIXED_USER_BW} Gbps")
    print(f"    - 总容量: {FIXED_NUM_LLMS * FIXED_LLM_CAPACITY} Gbps")
    print(
        f"    - 容量/需求比: {FIXED_NUM_LLMS * FIXED_LLM_CAPACITY / (FIXED_NUM_USERS * FIXED_USER_BW):.4f}"
    )


def main():
    """
    生成所有网络规模和分布组合的数据
    """
    print("=" * 60)
    print("固定16用户+4LLM实验数据生成器（所有分布组合）")
    print("=" * 60)
    print(f"用户数: {FIXED_NUM_USERS}")
    print(f"LLM数: {FIXED_NUM_LLMS}")
    print(f"用户需求: {FIXED_USER_BW} Gbps（固定）")
    print(f"LLM容量: {FIXED_LLM_CAPACITY} Gbps（固定）")
    print(f"总需求: {FIXED_NUM_USERS * FIXED_USER_BW} Gbps")
    print(f"总容量: {FIXED_NUM_LLMS * FIXED_LLM_CAPACITY} Gbps")
    print(
        f"容量/需求比: {FIXED_NUM_LLMS * FIXED_LLM_CAPACITY / (FIXED_NUM_USERS * FIXED_USER_BW):.4f}"
    )
    print(f"分布类型: {DISTRIBUTION_TYPES} （两两组合，共{len(DISTRIBUTION_TYPES)**2}种）")
    print(f"网络规模: {NETWORK_SIZES}")

    # 设置随机种子（确保可复现）
    random.seed(42)
    np.random.seed(42)

    # 为每个网络规模生成数据
    for network_size in NETWORK_SIZES:
        try:
            generate_network_for_size(network_size)
        except Exception as e:
            print(f"\n[错误] N-{network_size} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("所有数据生成完成！")
    print(f"{'='*60}")
    print(f"数据保存位置: sheets/N-<size>/distribution/")
    print(f"  - <user_dist>.xlsx (每个文件包含该user分布下所有llm分布的数据)")
    print(f"  - Sheet结构: 'user' sheet + 各个llm分布的sheet")
    print(f"  - node.xlsx (节点数据)")
    print(f"  - bandwidth.xlsx (带宽矩阵)")
    print(f"  - distance.xlsx (距离矩阵)")


if __name__ == '__main__':
    main()
