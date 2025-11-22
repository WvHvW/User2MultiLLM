import os
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import pandas as pd
from typing import List

import math
import random
import networkx as nx
import numpy as np


def generate_city_network(num_nodes=1000, target_degree=10):
    """
    生成一个稳健的全联通随机散点图拓扑。
    - 在整数坐标系中随机生成指定数量的节点
    - 节点坐标范围为(0, 100) x (0, 100)
    - 通过理论计算与后备机制保证网络全联通
    - 将 target_degree 视为期望的平均度
    """

    rng = np.random.default_rng()
    # 为了可视化和计算方便，仍在100x100的浮点坐标系中生成节点
    coords = {i: rng.uniform(0.0, 100.0, size=2) for i in range(num_nodes)}

    # --- 关键优化点 1: 基于理论计算连接半径 ---
    area = 100.0 * 100.0
    # 理论上的连通性保证半径
    connecting_radius = math.sqrt(area * math.log(num_nodes) /
                                  (math.pi * num_nodes))
    # 考虑目标度的半径
    target_radius = math.sqrt(target_degree * area / (math.pi * num_nodes))
    # 取两者中的较大值，并增加一个小的安全系数（例如1.1）以提高成功率
    radius = max(connecting_radius, target_radius) * 1.1

    # 生成随机几何图
    G = nx.random_geometric_graph(num_nodes, radius, pos=coords, dim=2)

    # 强制保证全联通
    if not nx.is_connected(G):
        # 获取所有的连通分量
        components = list(nx.connected_components(G))
        # 将所有分量连接到第一个分量上
        main_component = list(components[0])
        for i in range(1, len(components)):
            component = list(components[i])
            # 从每个分量中随机选择一个节点进行连接
            node_from_main = random.choice(main_component)
            node_from_other = random.choice(component)
            G.add_edge(node_from_main, node_from_other)

    nx.set_node_attributes(G, {n: tuple(v) for n, v in coords.items()}, 'pos')

    # 预先计算所有节点对之间的距离，便于后续调节平均度/度约束
    all_edge_candidates = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            all_edge_candidates.append((dist, i, j))
    all_edge_candidates.sort(key=lambda x: x[0])

    def _average_degree() -> float:
        return 2.0 * G.number_of_edges() / num_nodes if num_nodes else 0.0

    desired_avg = max(target_degree, 1.0)
    tolerance = 0.5
    min_degree = 3

    def _ensure_minimum_degree():
        """为所有节点补边以满足最小度约束。"""
        changed = False
        while True:
            low_degree_nodes = [
                node for node, deg in G.degree() if deg < min_degree
            ]
            if not low_degree_nodes:
                break

            node = low_degree_nodes[0]
            added = False
            for dist, u, v in all_edge_candidates:
                if node not in (u, v):
                    continue
                if G.has_edge(u, v):
                    continue
                G.add_edge(u, v)
                changed = True
                added = True
                break

            if not added:
                raise RuntimeError("无法满足最小度约束，请调整 target_degree 或节点规模")
        return changed

    _ensure_minimum_degree()

    # 若平均度偏低，按距离从短到长补边
    if _average_degree() < desired_avg - tolerance:
        for _, u, v in all_edge_candidates:
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v)
            if _average_degree() >= desired_avg - tolerance:
                break

    # 若平均度偏高，按距离从长到短删边，但保持连通
    if _average_degree() > desired_avg + tolerance:
        removal_candidates = [(float(np.linalg.norm(coords[u] - coords[v])), u,
                               v) for u, v in G.edges()]
        removal_candidates.sort(key=lambda x: x[0], reverse=True)

        for _, u, v in removal_candidates:
            if _average_degree() <= desired_avg + tolerance:
                break
            if not G.has_edge(u, v):
                continue
            if G.degree[u] <= min_degree or G.degree[v] <= min_degree:
                continue
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v)

    _ensure_minimum_degree()

    final_avg = _average_degree()
    if final_avg < desired_avg - 2 * tolerance or final_avg > desired_avg + 2 * tolerance:
        print(
            f"[WARN] 平均度 {final_avg:.2f} 与目标 {desired_avg:.2f} 差距较大，可调整 target_degree、节点数量或随机种子"
        )

    # 为边添加权重和容量属性
    for u, v in G.edges():
        pos_u = coords[u]
        pos_v = coords[v]
        distance = float(np.linalg.norm(pos_u - pos_v))
        cost = distance * 0.01
        G.edges[u, v]['distance'] = cost
        G.edges[u, v]['capacity_mbps'] = max(2500, (distance % 10) * 140)

    common_nodes = list(range(num_nodes))
    return G, common_nodes


def create_custom_network():
    """
    手动创建自定义网络拓扑，与 generate_city_network 输出格式完全兼容。

    使用方法：
        1. 修改 node_coords 字典，定义节点坐标
        2. 修改 edges 列表，定义连接关系
        3. 运行后自动生成距离和带宽矩阵

    返回:
        G: NetworkX 图对象，包含节点坐标、边权重（距离）和容量
        common_nodes: 节点ID列表
    """

    # ========== 在这里自定义你的网络 ==========

    # 节点坐标：{节点ID: (x坐标, y坐标)}
    # 坐标范围建议在 (0, 100) x (0, 100) 之间
    node_coords = {
        0: (50, 50),
        1: (40, 60),
        2: (40, 40),
        3: (60, 60),
        4: (60, 40),
        5: (30, 50),
        6: (20, 60),
        7: (20, 30),
        8: (50, 20),
        9: (30, 10),
        10: (40, 80),
        11: (50, 70),
        12: (70, 60),
        13: (70, 50),
        14: (80, 50),
        15: (80, 40),
        16: (70, 30),
        17: (70, 20),
        18: (50, 30),
        19: (70, 80)
    }

    # 边列表：[(节点1, 节点2), ...]
    # 定义哪些节点之间有连接
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 18), (1, 2), (1, 5), (1, 10),
             (2, 5), (2, 18), (3, 4), (3, 11), (3, 12), (4, 16), (4, 18),
             (5, 6), (5, 9), (6, 7), (6, 10), (7, 9), (8, 9), (8, 16), (8, 17),
             (10, 11), (11, 12), (11, 19), (12, 13), (12, 19), (13, 14),
             (13, 15), (14, 15), (14, 19), (15, 17), (16, 17), (16, 18)]

    # ========== 以下是自动处理逻辑，无需修改 ==========

    # 创建空的无向图
    G = nx.Graph()

    # 添加节点并设置坐标
    for node_id, (x, y) in node_coords.items():
        G.add_node(node_id, pos=(float(x), float(y)))

    # 添加边
    for u, v in edges:
        if u not in G.nodes or v not in G.nodes:
            raise ValueError(f"边 ({u}, {v}) 中的节点不存在于节点坐标字典中")
        G.add_edge(u, v)

    # 检查连通性
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"[警告] 网络不连通，共有 {len(components)} 个连通分量：")
        for i, comp in enumerate(components, 1):
            print(f"  分量 {i}: {sorted(comp)}")
        raise ValueError("网络必须是连通的。请添加更多边以连接所有节点。")

    # 为边添加权重和容量属性（与 generate_city_network 逻辑一致）
    for u, v in G.edges():
        pos_u = np.array(G.nodes[u]['pos'])
        pos_v = np.array(G.nodes[v]['pos'])
        distance = float(np.linalg.norm(pos_u - pos_v))

        # 距离成本（与 generate_city_network 一致：distance * 0.01）
        cost = distance * 0.01
        G.edges[u, v]['distance'] = cost

        # 带宽容量（与 generate_city_network 一致）
        # 使用距离的模运算生成变化的容量，最小500 Mbps
        G.edges[u, v]['capacity_mbps'] = max(500, (distance % 10) * 100)

    # 返回节点列表
    common_nodes = list(G.nodes())

    # 打印网络统计信息
    print(f"[自定义网络] 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    avg_degree = 2.0 * G.number_of_edges() / G.number_of_nodes()
    print(f"[自定义网络] 平均度: {avg_degree:.2f}")

    return G, common_nodes


def assign_user_nodes_by_distribution(G, distribution_type='uniform'):
    """
    根据指定的分布类型分配用户节点
    
    Parameters:
    G: NetworkX图对象
    distribution_type: 分布类型 ('uniform', 'sparse', 'gaussian', 'power_law')
    """

    # 获取所有节点ID
    all_nodes = list(G.nodes())

    # 重置所有节点的用户相关属性
    for n in all_nodes:
        # 清除可能存在的用户属性
        for attr in ['num_users', 'cpu_demand', 'mem_demand', 'bw_demand']:
            if attr in G.nodes[n]:
                del G.nodes[n][attr]

    # 定义用户节点的配置
    user_choices = [{
        'num': 60,
        'num_users': 100,
        'cpu_demand': 4,
        'mem_demand': 8,
        'bw_demand': 500
    }]

    # 计算总的用户节点数量
    total_user_nodes = sum(choice['num'] for choice in user_choices)

    # 检查是否有足够的节点
    if len(all_nodes) < total_user_nodes:
        raise ValueError(
            f"没有足够的节点。需要: {total_user_nodes}, 可用: {len(all_nodes)}")

    # 根据分布类型计算节点选择权重（基于实际坐标位置）
    if distribution_type == 'uniform':
        weights = np.ones(len(all_nodes))
    elif distribution_type == 'sparse':
        # 中心区域权重高，边缘区域权重低
        center_x, center_y = 50, 50
        weights = []
        for node_id in all_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            # 距离中心越近权重越高
            weight = np.exp(-distance_to_center / 30)  # 30是缩放因子
            weights.append(weight)
        weights = np.array(weights)
    elif distribution_type == 'gaussian':
        # 高斯分布，中心区域权重高
        center_x, center_y = 50, 50
        weights = []
        for node_id in all_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            # 高斯分布，标准差设为20
            weight = np.exp(-0.5 * (distance_to_center / 20)**2)
            weights.append(weight)
        weights = np.array(weights)
    elif distribution_type == 'power_law':
        # 幂律分布，中心区域权重高，边缘区域权重低
        center_x, center_y = 50, 50
        weights = []
        for node_id in all_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            # 避免除零错误，添加一个小常数
            weight = 1 / (distance_to_center + 1e-5)**1.5
            weights.append(weight)
        weights = np.array(weights)
    else:
        raise ValueError("不支持的分布类型")

    # 归一化权重
    weights = weights / np.sum(weights)

    available_indices = list(range(len(all_nodes)))

    def _sample_indices(quota: int) -> List[int]:
        if quota <= 0:
            return []
        if quota > len(available_indices):
            raise ValueError(
                f"没有足够的节点可供分配，需求 {quota}，剩余 {len(available_indices)}")
        selected: List[int] = []
        for _ in range(quota):
            if not available_indices:
                break
            weight_slice = np.array(
                [weights[idx] for idx in available_indices], dtype=float)
            total = weight_slice.sum()
            if total <= 0:
                weight_slice = np.full(len(available_indices),
                                       1.0 / len(available_indices))
            else:
                weight_slice /= total
            pick_pos = int(
                np.random.choice(len(available_indices), p=weight_slice))
            pick_index = int(available_indices.pop(pick_pos))
            selected.append(pick_index)
        return selected

    # 分配用户节点
    user_nodes: List[int] = []
    for choice in user_choices:
        selected_indices = _sample_indices(choice['num'])
        for idx in selected_indices:
            node_id = all_nodes[idx]
            user_nodes.append(node_id)
            # 设置用户相关属性
            G.nodes[node_id]['num_users'] = choice['num_users']
            G.nodes[node_id]['cpu_demand'] = choice['cpu_demand']
            G.nodes[node_id]['mem_demand'] = choice['mem_demand']
            G.nodes[node_id]['bw_demand'] = choice['bw_demand']
    return G, user_nodes


def assign_llm_nodes_by_distribution(G,
                                     distribution_type='uniform',
                                     user_nodes=None):
    """
    根据指定的分布类型分配LLM候选节点
    
    Parameters:
    G: NetworkX图对象
    distribution_type: 分布类型 ('uniform', 'sparse', 'gaussian', 'power_law')
    user_nodes: 已分配的用户节点列表，确保LLM节点与用户节点不重叠
    """

    # 获取所有节点ID
    all_nodes = list(G.nodes())

    # 重置所有节点的LLM相关属性
    for n in all_nodes:
        # 只有非用户节点才能被分配为LLM候选节点
        if user_nodes is None or n not in user_nodes:
            # 清除可能存在的LLM属性
            for attr in ['mem_capacity', 'cpu_capacity']:
                if attr in G.nodes[n]:
                    del G.nodes[n][attr]

    # 定义LLM候选节点的配置
    llm_choices = [{'num': 15, 'cpu_capacity': 18, 'mem_capacity': 16}]

    # 计算总的LLM节点数量
    total_llm_nodes = sum(choice['num'] for choice in llm_choices)

    # 确定可用于LLM分配的节点（排除用户节点）
    available_nodes = all_nodes[:]
    if user_nodes is not None:
        available_nodes = [n for n in all_nodes if n not in user_nodes]

    # 检查是否有足够的节点
    if len(available_nodes) < total_llm_nodes:
        raise ValueError(
            f"没有足够的非用户节点可供LLM分配。需要: {total_llm_nodes}, 可用: {len(available_nodes)}"
        )

    # 根据分布类型计算节点选择权重（基于实际坐标位置）
    if distribution_type == 'uniform':
        weights = np.ones(len(available_nodes))
    elif distribution_type == 'sparse':
        # 中心区域权重高，边缘区域权重低
        center_x, center_y = 50, 50
        weights = []
        for node_id in available_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            # 距离中心越远权重越高
            weight = np.exp(distance_to_center / 30)  # 30是缩放因子
            weights.append(weight)
        weights = np.array(weights)
    elif distribution_type == 'gaussian':
        # 高斯分布，中心区域权重高
        center_x, center_y = 50, 50
        weights = []
        for node_id in available_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            # 高斯分布，标准差设为20
            weight = np.exp(-0.5 * (distance_to_center / 20)**2)
            weights.append(weight)
        weights = np.array(weights)
    elif distribution_type == 'power_law':
        # 幂律分布，中心区域权重高，边缘区域权重低
        center_x, center_y = 50, 50
        weights = []
        for node_id in available_nodes:
            pos = G.nodes[node_id]['pos']
            distance_to_center = np.sqrt((pos[0] - center_x)**2 +
                                         (pos[1] - center_y)**2)
            # 避免除零错误，添加一个小常数
            weight = 1 / (distance_to_center + 1e-5)**1.5
            weights.append(weight)
        weights = np.array(weights)
    else:
        raise ValueError("不支持的分布类型")

    # 归一化权重
    weights = weights / np.sum(weights)

    available_indices = list(range(len(available_nodes)))

    def _sample_indices(quota: int) -> List[int]:
        if quota <= 0:
            return []
        if quota > len(available_indices):
            raise ValueError(
                f"没有足够的节点可供分配，需求 {quota}，剩余 {len(available_indices)}")
        selected: List[int] = []
        for _ in range(quota):
            if not available_indices:
                break
            weight_slice = np.array(
                [weights[idx] for idx in available_indices], dtype=float)
            total = weight_slice.sum()
            if total <= 0:
                weight_slice = np.full(len(available_indices),
                                       1.0 / len(available_indices))
            else:
                weight_slice /= total
            pick_pos = int(
                np.random.choice(len(available_indices), p=weight_slice))
            pick_index = int(available_indices.pop(pick_pos))
            selected.append(pick_index)
        return selected

    # 分配LLM候选节点
    llm_nodes: List[int] = []
    for choice in llm_choices:
        selected_indices = _sample_indices(choice['num'])
        for idx in selected_indices:
            node_id = available_nodes[idx]
            llm_nodes.append(node_id)
            # 设置LLM相关属性
            G.nodes[node_id]['cpu_capacity'] = choice['cpu_capacity']
            G.nodes[node_id]['mem_capacity'] = choice['mem_capacity']
            G.nodes[node_id]['deployed'] = 0
    return G, llm_nodes


# ========== 网络生成方式选择 ==========
# 设置为 True 使用自定义网络（修改 create_custom_network 函数内的坐标和边）
# 设置为 False 使用随机生成网络
USE_CUSTOM_NETWORK = False

# 生成基础网络拓扑
if USE_CUSTOM_NETWORK:
    print("=" * 50)
    print("使用自定义网络")
    print("=" * 50)
    Graph, common_nodes = create_custom_network()
else:
    print("=" * 50)
    print("使用随机生成网络")
    print("=" * 50)
    Graph, common_nodes = generate_city_network(num_nodes=150, target_degree=6)

# 根据不同分布类型分配节点并保存结果
distribution_types = ['uniform', 'power_law', 'sparse', 'gaussian']
user_data = {}
llm_data = {dist: {} for dist in distribution_types}

for user_dist in distribution_types:
    G = Graph.copy()
    G, current_user_nodes = assign_user_nodes_by_distribution(G, user_dist)
    user_info = []
    for node_id in current_user_nodes:
        node_data = G.nodes[node_id]
        info = {
            'node_id': node_id,
            'num_users': node_data.get('num_users', 0),
            'cpu_demand': node_data.get('cpu_demand', 0),
            'mem_demand': node_data.get('mem_demand', 0),
            'bw_demand': node_data.get('bw_demand', 0)
        }
        user_info.append(info)
    user_data[user_dist] = user_info
    for llm_dist in distribution_types:
        G_llm = G.copy()
        G_llm, current_llm_nodes = assign_llm_nodes_by_distribution(
            G_llm, llm_dist, user_nodes=current_user_nodes)
        llm_info = []
        for node_id in current_llm_nodes:
            node_data = G_llm.nodes[node_id]
            info = {
                'node_id': node_id,
                'cpu_capacity': node_data.get('cpu_capacity', 0),
                'mem_capacity': node_data.get('mem_capacity', 0),
            }
            llm_info.append(info)
        llm_data[user_dist][llm_dist] = llm_info

# 创建输出目录
os.makedirs(os.path.join('sheets', 'distribution'), exist_ok=True)
# 保存user、llm数据到Excel
for dist_type in distribution_types:
    user_file_path = os.path.join('sheets', 'distribution',
                                  f'{dist_type}.xlsx')
    with pd.ExcelWriter(user_file_path, engine='openpyxl') as writer:
        df = pd.DataFrame(user_data[dist_type])
        df.to_excel(writer, sheet_name=f"user", index=False)

        for llm_dist, data in llm_data[dist_type].items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f"{llm_dist}", index=False)
# 输出邻接矩阵到Excel
adjacency_matrix = pd.DataFrame(np.zeros(
    (len(Graph.nodes()), len(Graph.nodes()))),
                                index=list(Graph.nodes()),
                                columns=list(Graph.nodes()))
for u, v, attr in Graph.edges(data=True):
    adjacency_matrix.at[u, v] = 1
    adjacency_matrix.at[v, u] = 1
adjacency_matrix.to_excel('sheets/adjacency.xlsx')

# 输出节点数据到Excel (包含坐标)
node_data = []
for n, attr in Graph.nodes(data=True):
    node_info = {'node_id': n}
    node_info.update(attr)
    if 'pos' in node_info:
        node_info['pos_x'] = node_info['pos'][0]
        node_info['pos_y'] = node_info['pos'][1]
        del node_info['pos']  # 删除元组形式的pos，方便查看
    node_data.append(node_info)
node_df = pd.DataFrame(node_data)
node_df.to_excel('sheets/node.xlsx', index=False)

# 输出带宽矩阵到Excel
bandwidth_matrix = pd.DataFrame(np.zeros(
    (len(Graph.nodes()), len(Graph.nodes()))),
                                index=list(Graph.nodes()),
                                columns=list(Graph.nodes()))
for u, v, attr in Graph.edges(data=True):
    bandwidth_matrix.at[u, v] = attr['capacity_mbps']
    bandwidth_matrix.at[v, u] = attr['capacity_mbps']
bandwidth_matrix.to_excel('sheets/bandwidth.xlsx')

# 输出距离矩阵到Excel
distance_matrix = pd.DataFrame(np.zeros(
    (len(Graph.nodes()), len(Graph.nodes()))),
                               index=list(Graph.nodes()),
                               columns=list(Graph.nodes()))
for u, v, attr in Graph.edges(data=True):
    distance_matrix.at[u, v] = attr['distance']
    distance_matrix.at[v, u] = attr['distance']
distance_matrix.to_excel('sheets/distance.xlsx')

print("网络数据已生成并保存到 'sheets' 文件夹中。")
print("节点数据 (node.xlsx) 中已包含坐标 'pos_x' 和 'pos_y'。")
