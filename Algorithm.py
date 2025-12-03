"""
路由算法模块

根据ExperimentSetting.md实现9种路由算法：
1. no_split: 不分流算法（索引顺序）
2. no_split_aggregate: 不分流算法（带汇集图）
3. bottleneck_split: 瓶颈分流算法（带汇集图）
4. bottleneck_split_no_aggregate: 瓶颈分流算法（不带汇集图）
5. task_offloading: 任务卸载算法
6. task_offloading_aggregate: 任务卸载算法（带汇集图）
7. bottleneck_split_augment: 瓶颈分流增广算法
8. one_split: 1-split算法
9. one_split_augment: 1-split增广算法

所有算法返回统一格式的结果字典。
"""

import copy
from typing import Dict, List, Tuple, Any
from collections import deque


def _trace_user_flow_bfs(net, user_id, users: Dict, llms: Dict, S, T) -> List[Tuple]:
    """
    使用BFS追踪用户流量的最终分配情况

    从用户节点开始，遍历所有有流量的正向边，找到所有流向LLM的路径和流量。

    Args:
        net: 网络对象（增广完成后的最终状态）
        user_id: 用户ID
        users: 用户字典
        llms: LLM字典
        S: 超级源点ID
        T: 超级汇点ID

    Returns:
        [(user_id, llm_id, path, flow, cost), ...]
    """
    allocations = []

    # 如果用户不存在，返回空列表
    if user_id not in users:
        return allocations

    # 使用BFS找到所有从user到LLM的有流量路径
    # 我们需要找到所有的流量分配，可能有多条路径到不同LLM

    # 为了准确重建流量分配，我们使用DFS遍历所有可能的流量路径
    # 每找到一条完整路径（user -> ... -> LLM），记录该路径上的最小流量

    def dfs_find_paths(current, target_llms, path, visited_edges):
        """DFS找到从current到任意LLM的所有路径"""
        nonlocal allocations

        # 如果当前节点是LLM
        if current in target_llms:
            # 找到一条路径，计算该路径的流量和成本
            if len(path) >= 1:  # 至少有一条边
                # 计算路径上的最小流量
                min_flow = float('inf')
                for link in path:
                    min_flow = min(min_flow, link.flow)

                if min_flow > 1e-9:
                    # 计算路径成本（排除S->user和LLM->T的边）
                    path_nodes = [user_id]
                    path_cost = 0.0
                    for link in path:
                        if link.src != S and link.dst != T:
                            path_cost += link.flow * link.distance
                        path_nodes.append(link.dst)

                    # 去掉路径末尾的T节点（如果有）
                    if path_nodes[-1] == T:
                        path_nodes = path_nodes[:-1]

                    # 记录分配
                    allocations.append((user_id, current, path_nodes, min_flow, path_cost))

                    # 减少路径上的流量（虚拟减少，用于避免重复计数）
                    for link in path:
                        link.flow -= min_flow

            return

        # 继续DFS
        if current not in net.links:
            return

        for link in net.links[current]:
            # 只考虑正向边且有流量的边
            if link.is_reverse or link.flow < 1e-9:
                continue

            # 避免重复访问同一条边
            edge_id = (link.src, link.dst, id(link))
            if edge_id in visited_edges:
                continue

            # 跳过到S的边
            if link.dst == S:
                continue

            # 添加边到路径
            visited_edges.add(edge_id)
            path.append(link)

            # 递归DFS
            dfs_find_paths(link.dst, target_llms, path, visited_edges)

            # 回溯
            path.pop()
            visited_edges.remove(edge_id)

    # 创建网络的深拷贝，避免修改原网络
    net_copy = copy.deepcopy(net)

    # 多次DFS，直到没有更多流量
    max_iterations = 1000  # 防止无限循环
    for _ in range(max_iterations):
        # 检查是否还有从user出发的流量
        has_flow = False
        if user_id in net_copy.links:
            for link in net_copy.links[user_id]:
                if not link.is_reverse and link.flow > 1e-9:
                    has_flow = True
                    break

        if not has_flow:
            break

        # 执行一次DFS找到一条路径
        dfs_find_paths(user_id, llms.keys(), [], set())

    return allocations


def no_split(network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    不分流算法（索引顺序）

    策略：
    - 用户流量不分流，作为整体路由到单个LLM
    - 用户按索引顺序分配

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）

    Returns:
        {
            'allocations': [(user_id, llm_id, path, flow, cost), ...],
            'total_cost': float,
            'total_flow': float,
            'served_flow': float,
            'acceptance_ratio': float
        }
    """
    is_shared = kwargs.get('is_shared', True)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 按索引顺序处理用户
    sorted_users = sorted(users.items(), key=lambda x: x[0])

    for uid, user in sorted_users:
        demand = user.bw

        best_path = None
        best_llm = None
        best_cost = float('inf')

        for lid, llm in llms_copy.items():
            if is_shared and llm.service_capacity < demand:
                continue

            dist, prev = net.dijkstra_with_capacity(uid, demand, target_id=lid)

            if dist[lid] < best_cost:
                best_cost = dist[lid]
                best_llm = lid
                best_path = net.get_path(prev, uid, lid)

        if best_path and best_llm is not None:
            path_links = []
            for i in range(len(best_path) - 1):
                src, dst = best_path[i], best_path[i + 1]
                for link in net.links.get(src, []):
                    if link.dst == dst and not link.is_reverse:
                        path_links.append(link)
                        break

            net.send_flow(path_links, demand)

            if is_shared:
                llms_copy[best_llm].service_capacity -= demand

            path_cost = sum(link.distance for link in path_links) * demand
            allocations.append((uid, best_llm, best_path, demand, path_cost))
            total_cost += path_cost
            served_flow += demand

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def no_split_aggregate(network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    不分流算法（带汇集图）

    策略：
    - 所有用户连接到超源S，所有LLM连接到超汇T
    - 按当前最大用户节点的流量作为推流量，寻找S->T的最短路径，根据实际路径选择的用户推送流量
    - 当前无法找到路径时，踢出这个用户，使用第二大用户流量作为推流量推送

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）
            - epsilon_cost: float, S->user和LLM->T边的极小成本（默认1e-9）

    Returns:
        同no_split
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级源点S和超级汇点T
    S = -1
    T = -2
    import Entity
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 连接S到所有用户（单向边）
    for uid, user in users.items():
        net.add_one_way_link(S, uid, user.bw, epsilon_cost)

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 记录每个用户剩余流量
    user_remaining = {uid: user.bw for uid, user in users.items()}
    # 记录已踢出的用户
    excluded_users = set()

    while True:
        # 找到当前最大流量的用户（排除已踢出的）
        max_user = None
        max_demand = 0
        for uid, remaining in user_remaining.items():
            if uid not in excluded_users and remaining > 1e-9:
                if remaining > max_demand:
                    max_demand = remaining
                    max_user = uid

        if max_user is None:
            break  # 所有用户都已处理完

        # 寻找S->T的最短路径（推流量为max_demand）
        dist, prev = net.dijkstra_with_capacity(S, max_demand, target_id=T)

        if dist[T] >= float('inf'):
            # 无法找到路径，踢出这个用户
            excluded_users.add(max_user)
            continue

        # 重建路径
        path = net.get_path(prev, S, T)
        path_links = []
        bottleneck = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        # 识别路径中的用户和LLM（S->user->...->LLM->T）
        if len(path) < 4:  # 至少需要S->user->...->LLM->T
            excluded_users.add(max_user)
            continue

        actual_user = path[1]  # S后面的第一个节点是用户
        actual_llm = path[-2]  # T前面的节点是LLM

        # 验证是否是有效的用户和LLM
        if actual_user not in users or actual_llm not in llms_copy:
            excluded_users.add(max_user)
            continue

        # 推送流量（用户整个需求）
        user_demand = user_remaining[actual_user]
        push_flow = min(bottleneck, user_demand)

        if push_flow < user_demand - 1e-9:
            # 无法满足整个用户需求（no-split要求），踢出该用户
            excluded_users.add(actual_user)
            continue

        # 满足no-split约束，推送整个用户流量
        net.send_flow(path_links, user_demand)

        # 更新用户剩余流量
        user_remaining[actual_user] = 0

        # 记录分配（去掉S和T节点）
        path_without_ST = path[1:-1]
        # 计算成本（排除S->user和LLM->T边）
        path_cost = sum(
            link.distance for link in path_links[1:-1]
        ) * user_demand
        allocations.append((actual_user, actual_llm, path_without_ST, user_demand, path_cost))
        total_cost += path_cost
        served_flow += user_demand

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def bottleneck_split(network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流算法（带汇集图）

    策略：
    - 生成汇集图：S->users, LLMs->T（单向边，极小成本）
    - 使用链路瓶颈流量作为每一轮的推流粒度
    - 允许用户流量分流到不同LLM

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）
            - epsilon_cost: float, S->user和LLM->T边的极小成本（默认1e-9）

    Returns:
        同no_split
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级源点S和超级汇点T
    S = -1
    T = -2
    import Entity
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 连接S到所有用户（单向边）
    for uid, user in users.items():
        net.add_one_way_link(S, uid, user.bw, epsilon_cost)

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 循环增广，直到无可行路径
    while True:
        # 寻找S->T的最短路径（任意流量）
        dist, prev = net.dijkstra_with_capacity(S, float('inf'), target_id=T)

        if dist[T] >= float('inf'):
            break  # 无可行路径

        # 重建路径
        path = net.get_path(prev, S, T)
        path_links = []
        bottleneck = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        if bottleneck < 1e-9:
            break

        # 推送瓶颈流量
        net.send_flow(path_links, bottleneck)

        # 识别路径中的用户和LLM
        if len(path) >= 4:
            actual_user = path[1]
            actual_llm = path[-2]

            if actual_user in users and actual_llm in llms_copy:
                # 记录分配（去掉S和T节点）
                path_without_ST = path[1:-1]
                # 计算成本（排除S->user和LLM->T边）
                path_cost = sum(
                    link.distance for link in path_links[1:-1]
                ) * bottleneck
                allocations.append((actual_user, actual_llm, path_without_ST, bottleneck, path_cost))
                total_cost += path_cost
                served_flow += bottleneck

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def bottleneck_split_no_aggregate(network, users: Dict, llms: Dict, user_ideal_llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流算法（不带汇集图）

    策略：
    - 将user流量分流到user_ideal_llm前几个llm
    - 用户分配顺序默认索引顺序
    - 使用瓶颈流量作为推流粒度

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        user_ideal_llms: {user_id: {llm_id: cost, ...}} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）

    Returns:
        同no_split
    """
    is_shared = kwargs.get('is_shared', True)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 按索引顺序处理用户
    sorted_users = sorted(users.items(), key=lambda x: x[0])

    for uid, user in sorted_users:
        remaining = user.bw

        if uid not in user_ideal_llms:
            continue

        ideal_llms = user_ideal_llms[uid]
        sorted_ideal_llms = sorted(ideal_llms.items(), key=lambda x: x[1])

        # 循环分流，直到流量全部分配
        for lid, _ in sorted_ideal_llms:
            if remaining < 1e-9:
                break

            if lid not in llms_copy:
                continue

            if is_shared and llms_copy[lid].service_capacity < 1e-9:
                continue

            # 使用Dijkstra找路径
            dist, prev = net.dijkstra_with_capacity(uid, remaining, target_id=lid)

            if dist[lid] < float('inf'):
                path = net.get_path(prev, uid, lid)
                path_links = []
                bottleneck = float('inf')

                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    for link in net.links.get(src, []):
                        if link.dst == dst and not link.is_reverse:
                            path_links.append(link)
                            bottleneck = min(bottleneck, link.residual_capacity)
                            break

                # 考虑LLM容量限制
                if is_shared:
                    bottleneck = min(bottleneck, llms_copy[lid].service_capacity)

                # 推送瓶颈流量
                push_flow = min(bottleneck, remaining)

                if push_flow > 1e-9:
                    net.send_flow(path_links, push_flow)

                    if is_shared:
                        llms_copy[lid].service_capacity -= push_flow

                    path_cost = sum(link.distance for link in path_links) * push_flow
                    allocations.append((uid, lid, path, push_flow, path_cost))
                    total_cost += path_cost
                    served_flow += push_flow
                    remaining -= push_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def task_offloading(network, users: Dict, llms: Dict, user_ideal_llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    任务卸载算法

    策略：
    - 分流粒度为user_ideal_llm里最靠前的llm剩余容量
    - 用户分配顺序按默认索引顺序

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        user_ideal_llms: {user_id: {llm_id: cost, ...}} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）

    Returns:
        同no_split
    """
    is_shared = kwargs.get('is_shared', True)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 按索引顺序处理用户
    sorted_users = sorted(users.items(), key=lambda x: x[0])

    for uid, user in sorted_users:
        remaining = user.bw

        if uid not in user_ideal_llms:
            continue

        ideal_llms = user_ideal_llms[uid]
        sorted_ideal_llms = sorted(ideal_llms.items(), key=lambda x: x[1])

        for lid, _ in sorted_ideal_llms:
            if remaining < 1e-9:
                break

            if lid not in llms_copy:
                continue

            llm_capacity = llms_copy[lid].service_capacity if is_shared else float('inf')
            if llm_capacity < 1e-9:
                continue

            # 分流粒度为LLM剩余容量
            push_flow = min(llm_capacity, remaining)

            dist, prev = net.dijkstra_with_capacity(uid, push_flow, target_id=lid)

            if dist[lid] < float('inf'):
                path = net.get_path(prev, uid, lid)
                path_links = []
                bottleneck = float('inf')

                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    for link in net.links.get(src, []):
                        if link.dst == dst and not link.is_reverse:
                            path_links.append(link)
                            bottleneck = min(bottleneck, link.residual_capacity)
                            break

                # 实际推送流量受路径瓶颈限制
                actual_push = min(push_flow, bottleneck)

                if actual_push > 1e-9:
                    net.send_flow(path_links, actual_push)

                    if is_shared:
                        llms_copy[lid].service_capacity -= actual_push

                    path_cost = sum(link.distance for link in path_links) * actual_push
                    allocations.append((uid, lid, path, actual_push, path_cost))
                    total_cost += path_cost
                    served_flow += actual_push
                    remaining -= actual_push

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def task_offloading_aggregate(network, users: Dict, llms: Dict, user_ideal_llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    任务卸载算法（带汇集图）

    策略：
    - 所有用户连接到超源S，所有LLM连接到超汇T
    - 按当前最大用户节点的流量作为推流量，寻找S->T的最短路径，根据实际路径选择的用户推送流量
    - 推送流量的大小为用户剩余流量和llm剩余容量的最小值

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        user_ideal_llms: {user_id: {llm_id: cost, ...}} 字典（可选，不使用）
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）
            - epsilon_cost: float, S->user和LLM->T边的极小成本（默认1e-9）

    Returns:
        同no_split
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级源点S和超级汇点T
    S = -1
    T = -2
    import Entity
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 连接S到所有用户（单向边）
    for uid, user in users.items():
        net.add_one_way_link(S, uid, user.bw, epsilon_cost)

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 记录每个用户剩余流量
    user_remaining = {uid: user.bw for uid, user in users.items()}

    while True:
        # 找到当前最大流量的用户
        max_user = None
        max_demand = 0
        for uid, remaining in user_remaining.items():
            if remaining > 1e-9 and remaining > max_demand:
                max_demand = remaining
                max_user = uid

        if max_user is None:
            break

        # 寻找S->T的最短路径（推流量为max_demand）
        dist, prev = net.dijkstra_with_capacity(S, max_demand, target_id=T)

        if dist[T] >= float('inf'):
            break  # 无可行路径

        # 重建路径
        path = net.get_path(prev, S, T)
        path_links = []
        bottleneck = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        # 识别路径中的用户和LLM
        if len(path) < 4:
            break

        actual_user = path[1]
        actual_llm = path[-2]

        if actual_user not in users or actual_llm not in llms_copy:
            break

        # 推送流量的大小为用户剩余流量和llm剩余容量的最小值
        user_demand = user_remaining[actual_user]
        llm_capacity = llms_copy[actual_llm].service_capacity if is_shared else float('inf')
        push_flow = min(bottleneck, user_demand, llm_capacity)

        if push_flow < 1e-9:
            break

        net.send_flow(path_links, push_flow)

        # 更新用户剩余流量
        user_remaining[actual_user] -= push_flow

        # 记录分配（去掉S和T节点）
        path_without_ST = path[1:-1]
        # 计算成本（排除S->user和LLM->T边）
        path_cost = sum(
            link.distance for link in path_links[1:-1]
        ) * push_flow
        allocations.append((actual_user, actual_llm, path_without_ST, push_flow, path_cost))
        total_cost += path_cost
        served_flow += push_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def bottleneck_split_augment(network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流增广算法（带汇集图+负环消除）

    策略：
    - 生成汇集图：S->users, LLMs->T（单向边，极小成本）
    - 使用链路瓶颈流量作为每一轮的推流粒度
    - 每轮增广前消除负环
    - 每轮推流后通过BFS重建当前分配状态并记录

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）
            - epsilon_cost: float, S->user和LLM->T边的极小成本（默认1e-9）

    Returns:
        {
            'allocations': [...],  # 最终准确的分配结果
            'total_cost': float,  # 实际成本
            'round_allocations': [  # 每一轮的分配状态
                {
                    'round': int,
                    'allocations': [...],
                    'round_cost': float,
                    'cumulative_cost': float,
                    'round_flow': float,
                    'cumulative_flow': float
                },
                ...
            ],
            'total_flow': float,
            'served_flow': float,
            'acceptance_ratio': float
        }
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级源点S和超级汇点T
    S = -1
    T = -2
    import Entity
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 连接S到所有用户（单向边）
    for uid, user in users.items():
        net.add_one_way_link(S, uid, user.bw, epsilon_cost)

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    total_demand = sum(u.bw for u in users.values())

    # 记录每一轮的分配状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0

    # 循环增广
    while True:
        # 消除负环
        net.cancel_negative_cycles()

        # 寻找S->T的最短路径
        dist, prev = net.dijkstra_with_capacity(S, float('inf'), target_id=T)

        if dist[T] >= float('inf'):
            break

        # 重建路径
        path = net.get_path(prev, S, T)
        path_links = []
        bottleneck = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        if bottleneck < 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, bottleneck)

        # 每轮推流后，使用BFS重建当前的分配状态
        round_num += 1
        round_allocs = []
        round_cost = 0.0
        round_flow = 0.0

        for uid, user in users.items():
            user_allocations = _trace_user_flow_bfs(net, uid, users, llms_copy, S, T)
            round_allocs.extend(user_allocations)

            for _, _, _, flow, cost in user_allocations:
                round_flow += flow
                round_cost += cost

        cumulative_cost = round_cost
        cumulative_flow = round_flow

        # 记录本轮状态
        round_allocations.append({
            'round': round_num,
            'allocations': round_allocs,
            'round_cost': round_cost,
            'cumulative_cost': cumulative_cost,
            'round_flow': round_flow,
            'cumulative_flow': cumulative_flow
        })

    # 最终分配状态（最后一轮的结果）
    if round_allocations:
        final_round = round_allocations[-1]
        allocations = final_round['allocations']
        total_cost = final_round['cumulative_cost']
        served_flow = final_round['cumulative_flow']
    else:
        allocations = []
        total_cost = 0.0
        served_flow = 0.0

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def one_split(network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    1-split算法

    策略：
    - 生成汇集图：S->users, LLMs->T（单向边，极小成本）
    - 每一轮的推流粒度为1

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）
            - epsilon_cost: float, S->user和LLM->T边的极小成本（默认1e-9）

    Returns:
        同no_split
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级源点S和超级汇点T
    S = -1
    T = -2
    import Entity
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 连接S到所有用户（单向边）
    for uid, user in users.items():
        net.add_one_way_link(S, uid, user.bw, epsilon_cost)

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 循环增广，每次推流粒度为1
    while True:
        # 寻找S->T的最短路径（推流量为1）
        dist, prev = net.dijkstra_with_capacity(S, 1, target_id=T)

        if dist[T] >= float('inf'):
            break

        # 重建路径
        path = net.get_path(prev, S, T)
        path_links = []
        bottleneck = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        # 推流粒度为1
        push_flow = min(1, bottleneck)

        if push_flow < 1e-9:
            break

        net.send_flow(path_links, push_flow)

        # 识别路径中的用户和LLM
        if len(path) >= 4:
            actual_user = path[1]
            actual_llm = path[-2]

            if actual_user in users and actual_llm in llms_copy:
                # 记录分配（去掉S和T节点）
                path_without_ST = path[1:-1]
                # 计算成本（排除S->user和LLM->T边）
                path_cost = sum(
                    link.distance for link in path_links[1:-1]
                ) * push_flow
                allocations.append((actual_user, actual_llm, path_without_ST, push_flow, path_cost))
                total_cost += path_cost
                served_flow += push_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


def one_split_augment(network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    1-split增广算法（带负环消除）

    策略：
    - 生成汇集图：S->users, LLMs->T（单向边，极小成本）
    - 每一轮的推流粒度为1
    - 每轮增广前消除负环
    - 每轮推流后通过BFS重建当前分配状态并记录

    Args:
        network: Network对象
        users: {user_id: User对象} 字典
        llms: {llm_id: LLM对象} 字典
        **kwargs:
            - is_shared: bool, LLM容量是否共享（默认True）
            - epsilon_cost: float, S->user和LLM->T边的极小成本（默认1e-9）

    Returns:
        {
            'allocations': [...],  # 最终准确的分配结果
            'total_cost': float,  # 实际成本
            'round_allocations': [  # 每一轮的分配状态
                {
                    'round': int,
                    'allocations': [...],
                    'round_cost': float,
                    'cumulative_cost': float,
                    'round_flow': float,
                    'cumulative_flow': float
                },
                ...
            ],
            'total_flow': float,
            'served_flow': float,
            'acceptance_ratio': float
        }
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级源点S和超级汇点T
    S = -1
    T = -2
    import Entity
    net.add_node(Entity.Node(S, 0, 0))
    net.add_node(Entity.Node(T, 0, 0))

    # 连接S到所有用户（单向边）
    for uid, user in users.items():
        net.add_one_way_link(S, uid, user.bw, epsilon_cost)

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    total_demand = sum(u.bw for u in users.values())

    # 记录每一轮的分配状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0

    # 循环增广，每次推流粒度为1
    while True:
        # 消除负环
        net.cancel_negative_cycles()

        # 寻找S->T的最短路径（推流量为1）
        dist, prev = net.dijkstra_with_capacity(S, 1, target_id=T)

        if dist[T] >= float('inf'):
            break

        # 重建路径
        path = net.get_path(prev, S, T)
        path_links = []
        bottleneck = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        # 推流粒度为1
        push_flow = min(1, bottleneck)

        if push_flow < 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, push_flow)

        # 每轮推流后，使用BFS重建当前的分配状态
        round_num += 1
        round_allocs = []
        round_cost = 0.0
        round_flow = 0.0

        for uid, user in users.items():
            user_allocations = _trace_user_flow_bfs(net, uid, users, llms_copy, S, T)
            round_allocs.extend(user_allocations)

            for _, _, _, flow, cost in user_allocations:
                round_flow += flow
                round_cost += cost

        cumulative_cost = round_cost
        cumulative_flow = round_flow

        # 记录本轮状态
        round_allocations.append({
            'round': round_num,
            'allocations': round_allocs,
            'round_cost': round_cost,
            'cumulative_cost': cumulative_cost,
            'round_flow': round_flow,
            'cumulative_flow': cumulative_flow
        })

    # 最终分配状态（最后一轮的结果）
    if round_allocations:
        final_round = round_allocations[-1]
        allocations = final_round['allocations']
        total_cost = final_round['cumulative_cost']
        served_flow = final_round['cumulative_flow']
    else:
        allocations = []
        total_cost = 0.0
        served_flow = 0.0

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio
    }


# 算法注册表
ALGORITHMS = {
    'no-split': no_split,
    'no-split-aggregate': no_split_aggregate,
    'bottleneck-split': bottleneck_split,
    'bottleneck-split-no-aggregate': bottleneck_split_no_aggregate,
    'task-offloading': task_offloading,
    'task-offloading-aggregate': task_offloading_aggregate,
    'bottleneck-split-augment': bottleneck_split_augment,
    '1-split': one_split,
    '1-split-augment': one_split_augment,
}


def run_algorithm(algorithm_name: str, network, users: Dict, llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    运行指定的路由算法

    Args:
        algorithm_name: 算法名称
        network: Network对象
        users: 用户字典
        llms: LLM字典
        **kwargs: 算法特定参数

    Returns:
        算法结果字典

    Raises:
        ValueError: 如果算法名称不存在
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"未知算法: {algorithm_name}. 可用算法: {list(ALGORITHMS.keys())}")

    return ALGORITHMS[algorithm_name](network, users, llms, **kwargs)
