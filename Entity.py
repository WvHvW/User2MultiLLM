from collections import deque
import heapq
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

MIN_FLOW_EPSILON = 1e-9

script_dir = os.path.dirname(os.path.abspath(__file__))
sheets_dir = os.path.join(script_dir, 'sheets')


class Node:

    def __init__(self, id, pos_X, pos_Y):
        self.id = id
        self.role = 'common'
        self.pos_X = pos_X
        self.pos_Y = pos_Y


class User:

    def __init__(self, id, bw, computation, storage):
        self.id = id
        self.bw = bw
        self.computation = computation
        self.storage = storage


class LLM:

    def __init__(self, id, computation, storage):
        self.id = id
        self.computation = computation
        self.storage = storage
        self.available_computation = computation
        self.available_storage = storage


class Link:

    def __init__(self, src, dst, capacity, distance, is_reverse=False):
        # --- 现有属性保持不变 ---
        self.src = src
        self.dst = dst
        self.capacity = capacity
        self.distance = distance
        self.is_reverse = is_reverse
        self.flow = 0
        self.reverse = None

    @property
    def residual_capacity(self):
        if not self.is_reverse:
            return self.capacity - self.flow
        if self.reverse is None:
            return 0
        return self.reverse.flow


class Network:

    def __init__(self):
        self.nodes = {}
        self.links = {}
        # 节点势能（用于最小费用增广路径），在多次路由调用之间持久化
        self.node_potential = {}

    def add_node(self, node):
        self.nodes[node.id] = node
        self.links[node.id] = []

    def add_link(self, src, dst, capacity, cost):
        # 创建 src -> dst 的基础边和其残余边
        forward = Link(src, dst, capacity, cost, is_reverse=False)
        forward_residual = Link(dst, src, 0, cost, is_reverse=True)
        forward.reverse = forward_residual
        forward_residual.reverse = forward
        self.links.setdefault(src, []).append(forward)
        self.links.setdefault(dst, []).append(forward_residual)

        # 创建 dst -> src 的基础边和其残余边
        backward = Link(dst, src, capacity, cost, is_reverse=False)
        backward_residual = Link(src, dst, 0, cost, is_reverse=True)
        backward.reverse = backward_residual
        backward_residual.reverse = backward
        self.links.setdefault(dst, []).append(backward)
        self.links.setdefault(src, []).append(backward_residual)

    # 使用Dijkstra算法计算理想最短路径。
    def dijkstra_ideal(self, start_node_id, user_bw=1.0):

        distances = {node_id: float('inf') for node_id in self.nodes}
        previous_nodes = {
            node_id: None
            for node_id in self.nodes
        }  #用dict比list节省空间，毕竟不是所有节点都有前驱
        distances[start_node_id] = 0
        pq = [(0, start_node_id)]

        while pq:
            current_distance, current_node_id = heapq.heappop(pq)
            #基于贪心，每一步都是最短，后续的最短都基于这个最短，所以不会有更短的了
            if current_distance > distances[current_node_id]:
                continue
            if current_node_id in self.links:
                for link in self.links[current_node_id]:
                    neighbor_id = link.dst
                    distance = current_distance + link.distance
                    if distance < distances[neighbor_id]:
                        distances[neighbor_id] = distance
                        previous_nodes[neighbor_id] = current_node_id
                        heapq.heappush(pq, (distance, neighbor_id))

        costs = {
            node_id: distances[node_id] * user_bw
            for node_id in self.nodes
        }

        return distances, costs

    # 考虑链路容量的dijstra
    def dijkstra_with_capacity(self,
                               start_node_id,
                               min_capacity=0,
                               target_id=None):
        distances = {node_id: float('inf') for node_id in self.nodes}
        previous_nodes = {node_id: None for node_id in self.nodes}
        distances[start_node_id] = 0
        pq = [(0, start_node_id)]

        while pq:
            current_distance, current_node_id = heapq.heappop(pq)

            if target_id is not None and current_node_id == target_id:
                break

            if current_distance > distances[current_node_id]:
                continue

            if current_node_id in self.links:
                for link in self.links[current_node_id]:
                    # 只考虑满足最小容量要求的物理链路
                    if not link.is_reverse and link.capacity - link.flow >= min_capacity:
                        neighbor_id = link.dst
                        distance = current_distance + link.distance

                        if distance < distances[neighbor_id]:
                            distances[neighbor_id] = distance
                            previous_nodes[neighbor_id] = current_node_id
                            heapq.heappush(pq, (distance, neighbor_id))

            costs = {
                node_id: distances[node_id] * min_capacity
                for node_id in self.nodes
            }
        return distances, previous_nodes, costs

    def successive_shortest_paths(self,
                                  source: int,
                                  sink: int,
                                  max_flow: float,
                                  k: int = 1,
                                  trace: bool = False):
        """
        最小费用流：Successive Shortest Paths with reduced cost（节点势）

        参数:
            source: 源节点ID
            sink: 汇节点ID
            max_flow: 最大流量
            k: 每次增广的流量单位
            trace: 是否记录中介变量（用于调试和分析）

        返回:
            allocations: 分配结果列表，每条记录含 'path','cost','flow'
            如果trace=True，则返回 (allocations, trace_log)
        """
        import math, heapq
        INF = math.inf

        # 节点势（初始化或使用持久化的值）
        if not self.node_potential:
            # 第一次调用或重置后，初始化为 0
            self.node_potential = {nid: 0.0 for nid in self.nodes}

        pi = self.node_potential  # 直接使用持久化的势能引用

        allocations = []
        trace_log = [] if trace else None
        remaining = max_flow
        iteration = 0

        while remaining > 1e-9:
            iteration += 1
            push = min(k, remaining)

            # # 调试输出
            # if iteration % 10 == 1 or iteration <= 5:
            #     print(
            #         f"  [SSP] 迭代 {iteration}: remaining={remaining:.1f}, push={push:.1f}"
            #     )

            # # 接近完成时输出详细信息
            # if remaining <= 100:
            #     print(f"  [SSP] 迭代 {iteration}: remaining={remaining:.1f}, 进入Dijkstra...")

            # ---------- Dijkstra with reduced cost ----------
            dist = {nid: INF for nid in self.nodes}
            prev = {nid: None for nid in self.nodes}
            dist[source] = 0
            pq = [(0, source)]

            # 记录Dijkstra搜索的节点数（用于性能分析）
            nodes_explored = 0
            max_pq_size = 0
            EPSILON = 1e-9  # 浮点数精度保护

            while pq:
                # 保护1：限制优先队列大小，防止爆炸
                max_pq_size = max(max_pq_size, len(pq))
                if len(pq) > 10000:
                    print(f"  [SSP] 警告: 优先队列过大 ({len(pq)}), 可能存在负环")
                    break

                d, u = heapq.heappop(pq)
                if d > dist[u] + EPSILON:  # 保护2：浮点精度容差
                    continue

                nodes_explored += 1

                # 保护3：限制探索节点数，防止死循环
                if nodes_explored > len(self.nodes) * 100:
                    print(f"  [SSP] 警告: 探索节点过多 ({nodes_explored}), 终止Dijkstra")
                    break

                for link in self.links.get(u, []):
                    rc = link.residual_capacity
                    # 只要有残余容量就考虑，不强制要求 >= push
                    if rc <= EPSILON:  # 浮点精度保护
                        continue
                    v = link.dst
                    # reduced cost（允许使用由残余边构成的负环）
                    raw_cost = (link.distance if not link.is_reverse else
                                -link.reverse.distance)
                    cost = raw_cost + pi[u] - pi[v]

                    # 保护4：只有显著改进才松弛，避免微小浮点差异导致反复入队
                    if dist[u] + cost < dist[v] - EPSILON:
                        dist[v] = dist[u] + cost
                        prev[v] = (u, link)
                        heapq.heappush(pq, (dist[v], v))

            # # 接近完成时输出 Dijkstra 统计信息
            # if remaining <= 100:
            #     print(f"  [SSP] Dijkstra完成: nodes_explored={nodes_explored}, max_pq_size={max_pq_size}, dist[sink]={'INF' if dist[sink]==INF else f'{dist[sink]:.2f}'}")

            # 如果此轮无法从 source 到 sink 找到可行增广路，
            # 直接终止循环，剩余未分配流量视为舍弃
            if dist[sink] == INF:
                print(f"  [SSP] 迭代 {iteration}: 无增广路, 舍弃剩余流量 {remaining:.1f}")
                if trace:
                    trace_log.append({
                        'iteration': iteration,
                        'status': 'no_augmenting_path',
                        'remaining_flow': remaining,
                        'push_amount': push,
                        'nodes_explored': nodes_explored
                    })
                break

            # ---------- 提取路径 ----------
            node_path = []
            link_path = []
            cur = sink
            while cur != source:
                node_path.append(cur)
                prev_node, prev_link = prev[cur]
                link_path.append(prev_link)
                cur = prev_node
            node_path.append(source)
            node_path.reverse()
            link_path.reverse()

            link_details = []
            if trace:
                for lk in link_path:
                    link_details.append({
                        'src': lk.src,
                        'dst': lk.dst,
                        'is_reverse': lk.is_reverse,
                        'capacity': lk.capacity,
                        'flow_before': lk.flow,
                        'residual_capacity': lk.residual_capacity,
                        'distance': lk.distance
                    })

            # ---------- 真正推流 ----------
            reduced_cost_path = dist[sink]
            self.send_flow(link_path, push)
            real_cost = (reduced_cost_path + pi[sink] - pi[source]) * push

            # # 接近完成时输出推流信息
            # if remaining <= 100:
            #     print(f"  [SSP] 推流完成: path_len={len(node_path)}, real_cost={real_cost:.2f}")

            allocations.append({
                "algorithm": f"{k}-split-augment",
                "user_id": node_path[1],  # 超源后第一跳即用户
                "llm_id": node_path[-2],  # 超汇前最后一跳即LLM
                "path": node_path[1:-1],
                "cost": real_cost,
                "flow": push
            })
            remaining -= push

            # ---------- 记录中介变量 ----------
            if trace:
                trace_log.append({
                    'iteration': iteration,
                    'status': 'success',
                    'push_amount': push,
                    'remaining_after': remaining,
                    'node_path': node_path,
                    'reduced_cost': reduced_cost_path,
                    'real_cost': real_cost,
                    'node_potentials_before': pi.copy(),
                    'link_details': link_details,
                    'nodes_explored': nodes_explored
                })

            # ---------- 更新节点势 ----------
            updated_count = 0
            for nid in self.nodes:
                if dist[nid] < INF:
                    pi[nid] += dist[nid]
                    updated_count += 1

            # # 接近完成时输出势能更新信息
            # if remaining <= 100:
            #     print(
            #         f"  [SSP] 势能更新完成: updated_nodes={updated_count}/{len(self.nodes)}, 迭代{iteration}结束\n"
            #     )

        if trace:
            return allocations, trace_log
        return allocations

    def _find_flow_path(self, source, sink, remaining_flow):
        queue = deque([source])
        parents = {source: None}

        while queue:
            current = queue.popleft()
            if current == sink:
                break

            for link in self.links.get(current, []):
                if link.is_reverse:
                    continue
                if remaining_flow.get(link, 0.0) <= MIN_FLOW_EPSILON:
                    continue
                next_node = link.dst
                if next_node in parents:
                    continue
                parents[next_node] = (current, link)
                queue.append(next_node)

        if sink not in parents:
            return []

        path_links = []
        node = sink
        while node != source:
            prev_node, link = parents[node]
            path_links.append(link)
            node = prev_node
        path_links.reverse()
        return path_links

    def decompose_flow_paths(self, source, sink, algorithm_name):
        remaining_flow = {}
        for link_list in self.links.values():
            for link in link_list:
                if link.is_reverse:
                    continue
                if link.flow > MIN_FLOW_EPSILON:
                    remaining_flow[link] = float(link.flow)

        allocations = []

        while True:
            path_links = self._find_flow_path(source, sink, remaining_flow)
            if not path_links:
                break

            min_flow = min(remaining_flow[link] for link in path_links)
            if min_flow <= MIN_FLOW_EPSILON:
                break

            node_path = [source]
            for link in path_links:
                node_path.append(link.dst)

            if len(node_path) < 4:
                break

            user_id = node_path[1]
            llm_id = node_path[-2]
            unit_cost = sum(link.distance for link in path_links)

            allocations.append({
                "algorithm": algorithm_name,
                "user_id": user_id,
                "llm_id": llm_id,
                "path": node_path[1:-1],
                "cost": unit_cost * min_flow,
                "flow": min_flow
            })

            for link in path_links:
                remaining_flow[link] -= min_flow
                if remaining_flow[link] <= MIN_FLOW_EPSILON:
                    del remaining_flow[link]

        return allocations

    def get_path(self, previous_nodes, start_node_id, target_node_id):
        path = []
        current_id = target_node_id
        while current_id is not None:
            path.insert(0, current_id)  #因为是反向追踪，所以插入到开头
            current_id = previous_nodes.get(current_id)
        if path and path[0] == start_node_id:
            return path
        return []

    def get_path_with_links(self, previous_nodes, start_node_id,
                            target_node_id):
        node_path = self.get_path(previous_nodes, start_node_id,
                                  target_node_id)
        if not node_path:
            return [], []

        path_links = []
        for src, dst in zip(node_path[:-1], node_path[1:]):
            link = next((lk for lk in self.links.get(src, [])
                         if not lk.is_reverse and lk.dst == dst), None)
            if link is None:
                raise ValueError(
                    f"Missing physical link for segment {src}->{dst}")
            path_links.append(link)

        return node_path, path_links

    def send_flow(self, path_links, flow_amount):
        for link in path_links:
            if not link.is_reverse:
                link.flow += flow_amount
            else:
                link.reverse.flow -= flow_amount

    def backtrack_flow(self, path_links, flow_amount):
        for link in path_links:
            if not link.is_reverse:
                link.flow -= flow_amount
            else:
                link.reverse.flow += flow_amount

    def reset_network(self, llms):
        for link_list in self.links.values():
            for link in link_list:
                link.flow = 0

        for llm in llms.values():
            llm.available_computation = llm.computation
            llm.available_storage = llm.storage

        # 重置节点势能，确保下次调用SSP时从干净状态开始
        self.node_potential.clear()


def _load_excel(path, sheet_name=0, index_col=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, index_col=index_col)


def load_network_from_sheets(sheets_root=sheets_dir):
    """根据 sheets 中的表格装配网络、节点、用户与 LLM 信息。"""

    adjacency_path = os.path.join(sheets_root, 'adjacency.xlsx')
    bandwidth_path = os.path.join(sheets_root, 'bandwidth.xlsx')
    distance_path = os.path.join(sheets_root, 'distance.xlsx')
    node_path = os.path.join(sheets_root, 'node.xlsx')

    bw_df = _load_excel(bandwidth_path, index_col=0)
    dist_df = _load_excel(distance_path, index_col=0)
    node_df = _load_excel(node_path)
    adj_df = _load_excel(adjacency_path, index_col=0)
    bw_df = bw_df.fillna(0)
    dist_df = dist_df.fillna(0)
    node_df = node_df.fillna(0)

    # network
    network = Network()
    node_entities = {}

    for row in node_df.itertuples(index=False):
        node = Node(id=int(row.node_id),
                    pos_X=float(row.pos_x),
                    pos_Y=float(row.pos_y))
        network.add_node(node)
        node_entities[node.id] = node

    # 只遍历上三角，避免重复添加
    for i in adj_df.index:
        for j in adj_df.columns:
            if int(i) >= int(j):
                continue
            if int(adj_df.loc[i, j]) == 0:
                continue
            capacity = float(bw_df.loc[i, j])
            distance = float(dist_df.loc[i, j])
            if capacity <= 0 or distance <= 0:
                continue
            network.add_link(int(i), int(j), capacity, distance)

    return {'network': network, 'nodes': node_entities}


def load_llm_info(user_distribution='uniform',
                  llm_distribution='uniform',
                  sheets_root=sheets_dir):
    """加载 LLM 信息。"""
    llm_path = os.path.join(sheets_root, 'distribution',
                            f'{user_distribution}.xlsx')
    llm_df = _load_excel(llm_path, sheet_name=llm_distribution)
    llms = {}
    for row in llm_df.itertuples(index=False):
        llm = LLM(id=int(row.node_id),
                  computation=float(row.cpu_capacity),
                  storage=float(row.mem_capacity))
        llms[llm.id] = llm
    return llms


def load_user_info(distribution='uniform', sheets_root=sheets_dir):
    """加载用户信息。"""
    user_path = os.path.join(sheets_root, 'distribution',
                             f'{distribution}.xlsx')
    user_df = _load_excel(user_path, sheet_name='user')
    users = {}
    for row in user_df.itertuples(index=False):
        user = User(id=int(row.node_id),
                    bw=float(row.bw_demand),
                    computation=float(row.cpu_demand),
                    storage=float(row.mem_demand))
        users[user.id] = user
    return users


def visualize_network(nodes,
                      network,
                      llms,
                      users,
                      user_distribution='uniform',
                      llm_distribution='uniform',
                      algorithm='no_split'):
    """
    可视化网络结构，包括节点角色、LLM候选节点和用户节点
    
    Parameters:
    nodes: 节点字典 {node_id: Node}
    network: Network对象
    llms: LLM字典
    users: 用户字典
    """
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    node_sizes = []
    node_labels = {}

    color_map = {
        'core': 'red',
        'agg': 'blue',
        'common': 'green',
        'candidate': 'orange',
        'llm': 'pink',
        'user': 'purple'
    }
    size_map = {
        'core': 500,
        'agg': 300,
        'common': 150,
        'candidate': 200,
        'llm': 200,
        'user': 180
    }

    # 确定每个节点的最终角色
    node_roles = {node.id: node.role for node_id, node in enumerate(nodes)}
    node_roles.update({llm_id: 'llm' for llm_id in llms})
    node_roles.update({user_id: 'user' for user_id in users})

    # 添加所有节点到图，并设置其属性和坐标
    for node_id, node in enumerate(nodes):
        role = node_roles.get(node_id, 'common')
        G.add_node(node_id)
        pos[node_id] = (node.pos_X, node.pos_Y)
        node_colors.append(color_map.get(role, 'gray'))
        node_sizes.append(size_map.get(role, 100))
        node_labels[node_id] = f"{node_id}\n{role.capitalize()}"

    # 添加边
    added_edges = set()
    edge_attrs = {}
    for src_node_id, links in network.links.items():
        for link in links:
            if link.is_reverse:
                continue
            edge_key = (src_node_id, link.dst)
            if edge_key in added_edges:
                continue
            G.add_edge(src_node_id, link.dst)
            added_edges.add(edge_key)
            edge_attrs[edge_key] = (link.distance, link.capacity)
    # 绘制图形
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G,
                           pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9)
    nx.draw_networkx_edges(G,
                           pos,
                           width=1.0,
                           alpha=0.5,
                           edge_color='gray',
                           arrows=True,
                           arrowstyle='-|>',
                           arrowsize=12)
    nx.draw_networkx_labels(G,
                            pos,
                            node_labels,
                            font_size=8,
                            font_color='black')

    # 绘制边的容量标注
    edge_labels = {}
    for u, v in G.edges():
        attrs = edge_attrs.get((u, v), edge_attrs.get((v, u)))
        if attrs is None:
            continue
        distance, capacity = attrs
        edge_labels[(u, v)] = f"{distance:.2f}/\n{capacity:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0],
                   marker='o',
                   color='w',
                   label=role.capitalize(),
                   markerfacecolor=color,
                   markersize=10) for role, color in color_map.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(
        f"Network Topology ({user_distribution} users, {llm_distribution} LLMs)",
        fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # 确保可视化文件夹存在
    os.makedirs('visualization', exist_ok=True)
    filename = f'visualization/{algorithm}-{user_distribution}-{llm_distribution}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()  # 关闭图像，避免在Jupyter等环境中连续显示


DISTRIBUTION_TYPES = ['uniform', 'sparse', 'gaussian', 'power_law']
