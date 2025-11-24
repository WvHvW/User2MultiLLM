"""
大规模网络生成器（支持万级节点）

优化策略：
1. 使用KD-Tree加速最近邻查询，避免O(n²)预计算
2. 简化度调整逻辑，只保证连通性和最小度约束
3. 按需计算距离，不预存储所有节点对
4. 输出稀疏边列表格式，避免生成亿级单元格的密集矩阵
"""

import os
import math
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import random

# 导入PN.py的分布生成函数
import sys
sys.path.append(os.path.dirname(__file__))
from PN import assign_user_nodes_by_distribution, assign_llm_nodes_by_distribution


def generate_large_city_network(num_nodes=10000, target_degree=10, min_degree=3):
    """
    生成大规模全连通随机几何网络（优化版，支持万级节点）

    参数:
        num_nodes: 节点数量
        target_degree: 目标平均度（作为连接半径的参考）
        min_degree: 最小度约束

    返回:
        G: NetworkX图对象
        common_nodes: 节点ID列表
    """
    print(f"开始生成 {num_nodes} 节点的大规模网络...")

    rng = np.random.default_rng()
    coords = {i: rng.uniform(0.0, 100.0, size=2) for i in range(num_nodes)}

    # 将坐标转换为数组用于KD-Tree
    coords_array = np.array([coords[i] for i in range(num_nodes)])

    # 构建KD-Tree加速最近邻查询
    print("构建KD-Tree索引...")
    tree = KDTree(coords_array)

    # 计算连接半径（基于理论连通性保证）
    area = 100.0 * 100.0
    connecting_radius = math.sqrt(area * math.log(num_nodes) / (math.pi * num_nodes))
    target_radius = math.sqrt(target_degree * area / (math.pi * num_nodes))
    radius = max(connecting_radius, target_radius) * 1.2  # 增加安全系数

    print(f"连接半径: {radius:.2f}")

    # 使用KD-Tree范围查询构建图
    print("构建初始边...")
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    edge_count = 0
    for i in range(num_nodes):
        if i % 1000 == 0:
            print(f"  处理节点 {i}/{num_nodes}, 当前边数: {edge_count}")

        # 查询半径内的所有邻居
        neighbors = tree.query_ball_point(coords_array[i], radius)
        for j in neighbors:
            if i < j:  # 避免重复边
                G.add_edge(i, j)
                edge_count += 1

    print(f"初始边数: {G.number_of_edges()}, 平均度: {2*G.number_of_edges()/num_nodes:.2f}")

    # 保证连通性
    print("检查连通性...")
    if not nx.is_connected(G):
        print("  网络未连通，强制连接所有分量...")
        components = list(nx.connected_components(G))
        main_component = list(components[0])

        for i in range(1, len(components)):
            component = list(components[i])
            # 找到两个分量之间距离最近的节点对
            min_dist = float('inf')
            best_pair = None

            for node_main in main_component[:min(100, len(main_component))]:  # 限制搜索范围
                for node_comp in component[:min(100, len(component))]:
                    dist = np.linalg.norm(coords_array[node_main] - coords_array[node_comp])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (node_main, node_comp)

            if best_pair:
                G.add_edge(best_pair[0], best_pair[1])
                main_component.extend(component)

    print("网络已连通")

    # 保证最小度约束
    print("保证最小度约束...")
    iteration = 0
    while iteration < 5:  # 最多迭代5次，避免死循环
        low_degree_nodes = [node for node, deg in G.degree() if deg < min_degree]
        if not low_degree_nodes:
            break

        print(f"  迭代 {iteration+1}: {len(low_degree_nodes)} 个节点度数 < {min_degree}")

        for node in low_degree_nodes[:min(1000, len(low_degree_nodes))]:  # 批量处理
            # 使用KD-Tree找到最近的k个邻居
            k_neighbors = 20  # 查找20个最近邻居
            distances, indices = tree.query(coords_array[node], k=k_neighbors+1)

            for neighbor in indices[1:]:  # 跳过自己（索引0）
                if not G.has_edge(node, neighbor):
                    G.add_edge(node, neighbor)
                    if G.degree[node] >= min_degree:
                        break

        iteration += 1

    final_low = [node for node, deg in G.degree() if deg < min_degree]
    if final_low:
        print(f"  [警告] 仍有 {len(final_low)} 个节点度数 < {min_degree}")

    print(f"最终边数: {G.number_of_edges()}, 平均度: {2*G.number_of_edges()/num_nodes:.2f}")

    # 设置节点坐标属性
    nx.set_node_attributes(G, {n: tuple(coords[n]) for n in range(num_nodes)}, 'pos')

    # 为边添加权重和容量属性
    print("计算边属性...")
    for u, v in G.edges():
        pos_u = coords[u]
        pos_v = coords[v]
        distance = float(np.linalg.norm(pos_u - pos_v))
        cost = distance * 0.01
        G.edges[u, v]['distance'] = cost
        G.edges[u, v]['capacity_mbps'] = max(2500, (distance % 10) * 140)

    print("网络生成完成！")
    common_nodes = list(range(num_nodes))
    return G, common_nodes


def save_network_to_sheets(Graph, output_dir='sheets'):
    """
    将网络保存为稀疏边列表格式（与Entity.py兼容）

    参数:
        Graph: NetworkX图对象
        output_dir: 输出目录
    """
    print(f"保存网络数据到 {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)

    # 输出节点数据（包含坐标）
    print("  保存节点数据...")
    node_data = []
    for n, attr in Graph.nodes(data=True):
        node_info = {'node_id': n}
        if 'pos' in attr:
            node_info['pos_x'] = attr['pos'][0]
            node_info['pos_y'] = attr['pos'][1]
        node_data.append(node_info)

    node_df = pd.DataFrame(node_data)
    node_df.to_excel(os.path.join(output_dir, 'node.xlsx'), index=False)

    # 输出边列表（稀疏格式）
    print("  保存边列表（稀疏格式）...")
    edge_list = []
    for u, v, attr in Graph.edges(data=True):
        edge_list.append({
            'src': u,
            'dst': v,
            'distance': attr.get('distance', 0),
            'capacity_mbps': attr.get('capacity_mbps', 0)
        })

    edge_df = pd.DataFrame(edge_list)
    edge_df.to_excel(os.path.join(output_dir, 'edge_list.xlsx'), index=False)

    print(f"  节点文件: {len(node_data)} 节点")
    print(f"  边列表文件: {len(edge_list)} 条边（稀疏格式）")
    print("网络数据保存完成！")


if __name__ == "__main__":
    # 生成10000节点网络
    num_nodes = 10000
    target_degree = 10

    Graph, common_nodes = generate_large_city_network(
        num_nodes=num_nodes,
        target_degree=target_degree,
        min_degree=3
    )

    # 保存基础网络结构到sheets目录
    save_network_to_sheets(Graph, output_dir='sheets')

    # 生成用户和LLM分布数据（与PN.py一致）
    print("\n生成用户和LLM分布数据...")
    distribution_types = ['uniform', 'power_law', 'sparse', 'gaussian', 'poisson']
    user_data = {}
    llm_data = {dist: {} for dist in distribution_types}

    for user_dist in distribution_types:
        print(f"  处理用户分布: {user_dist}")
        G = Graph.copy()
        G, current_user_nodes = assign_user_nodes_by_distribution(G, user_dist)
        user_info = []
        for node_id in current_user_nodes:
            node_data_dict = G.nodes[node_id]
            info = {
                'node_id': node_id,
                'num_users': node_data_dict.get('num_users', 0),
                'cpu_demand': node_data_dict.get('cpu_demand', 0),
                'mem_demand': node_data_dict.get('mem_demand', 0),
                'bw_demand': node_data_dict.get('bw_demand', 0)
            }
            user_info.append(info)
        user_data[user_dist] = user_info

        for llm_dist in distribution_types:
            print(f"    处理LLM分布: {user_dist}-{llm_dist}")
            G_llm = G.copy()
            G_llm, current_llm_nodes = assign_llm_nodes_by_distribution(
                G_llm, llm_dist, user_nodes=current_user_nodes)
            llm_info = []
            for node_id in current_llm_nodes:
                node_data_dict = G_llm.nodes[node_id]
                info = {
                    'node_id': node_id,
                    'cpu_capacity': node_data_dict.get('cpu_capacity', 0),
                    'mem_capacity': node_data_dict.get('mem_capacity', 0),
                }
                llm_info.append(info)
            llm_data[user_dist][llm_dist] = llm_info

    # 保存分布数据到sheets/distribution/
    print("\n保存分布数据...")
    os.makedirs(os.path.join('sheets', 'distribution'), exist_ok=True)
    for dist_type in distribution_types:
        user_file_path = os.path.join('sheets', 'distribution',
                                      f'{dist_type}.xlsx')
        with pd.ExcelWriter(user_file_path, engine='openpyxl') as writer:
            df = pd.DataFrame(user_data[dist_type])
            df.to_excel(writer, sheet_name=f"user", index=False)

            for llm_dist, data in llm_data[dist_type].items():
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=f"{llm_dist}", index=False)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("网络生成完成！")
    print(f"  节点数: {Graph.number_of_nodes()}")
    print(f"  边数: {Graph.number_of_edges()}")
    print(f"  平均度: {2 * Graph.number_of_edges() / Graph.number_of_nodes():.2f}")
    print(f"  是否连通: {nx.is_connected(Graph)}")

    degrees = [deg for _, deg in Graph.degree()]
    print(f"  最小度: {min(degrees)}")
    print(f"  最大度: {max(degrees)}")
    print(f"  度中位数: {np.median(degrees):.2f}")
    print(f"\n输出文件:")
    print(f"  sheets/node.xlsx - 节点坐标")
    print(f"  sheets/edge_list.xlsx - 边列表（稀疏格式）")
    print(f"  sheets/distribution/*.xlsx - 用户和LLM分布数据")
    print("=" * 50)
