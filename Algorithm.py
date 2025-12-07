"""
路由算法模块

根据ExperimentSetting.md实现10种路由算法：
1. no_split: 不分流算法（索引顺序）
2. no_split_aggregate: 不分流算法（带汇集图）
3. bottleneck_split: 瓶颈分流算法（带汇集图）
4. bottleneck_split_no_aggregate: 瓶颈分流算法（不带汇集图）
5. LLM_split: LLM分流算法
6. LLM_split_aggregate: LLM分流算法（带汇集图）
7. bottleneck_split_augment: 瓶颈分流增广算法
8. one_split: 1-split算法
9. one_split_no_aggregate: 1-split算法（不带汇集图）
10. one_split_augment: 1-split增广算法

所有算法返回统一格式的结果字典。
"""

import copy
from typing import Dict, List, Tuple, Any
from collections import deque


def _trace_user_flow_bfs(net, user_id, users: Dict, llms: Dict, S,
                         T) -> List[Tuple]:
    """
    使用迭代式DFS追踪用户流量的最终分配情况
    从用户节点开始，遍历所有有流量的正向边，找到所有流向LLM的路径和流量。
    """
    allocations = []

    # 如果用户不存在，返回空列表
    if user_id not in users:
        return allocations

    # 创建网络的深拷贝，避免修改原网络
    net_copy = copy.deepcopy(net)

    target_llms = set(llms.keys())

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

        # 使用迭代式DFS找到一条路径（避免递归栈溢出）
        # 栈中每个元素：(当前节点, 路径边列表, 已访问边集合)
        stack = [(user_id, [], set())]
        found_path = False

        while stack and not found_path:
            current, path, visited_edges = stack.pop()

            # 如果当前节点是LLM，找到一条路径
            if current in target_llms:
                if len(path) >= 1:
                    # 计算路径上的最小流量
                    min_flow = min(link.flow for link in path)

                    if min_flow > 1e-9:
                        # 计算路径成本（排除S->user和LLM->T的边）
                        path_nodes = [user_id]
                        path_distance = 0.0
                        for link in path:
                            if link.src != S and link.dst != T:
                                path_distance += link.distance
                            path_nodes.append(link.dst)

                        # 去掉路径末尾的T节点（如果有）
                        if path_nodes[-1] == T:
                            path_nodes = path_nodes[:-1]

                        # 总成本 = 单位距离 × 流量
                        path_cost = path_distance * min_flow

                        # 记录分配
                        allocations.append((user_id, current, path_nodes,
                                            min_flow, path_cost))

                        # 减少路径上的流量（避免重复计数）
                        for link in path:
                            link.flow -= min_flow

                        found_path = True
                continue

            # 继续DFS：遍历当前节点的所有出边
            if current not in net_copy.links:
                continue

            for link in net_copy.links[current]:
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

                # 创建新的路径和访问集合（避免共享状态）
                new_path = path + [link]
                new_visited = visited_edges | {edge_id}

                # 压入栈（后进先出，实现DFS）
                stack.append((link.dst, new_path, new_visited))

    return allocations


def no_split(network, users: Dict, llms: Dict, user_ideal_llms: Dict,
             **kwargs) -> Dict[str, Any]:
    """
    不分流算法（不带汇集图）

    策略：
    - 用户流量不分流，作为整体路由到单个LLM
    - 每个用户分配给其user_ideal_llms中第一个可用的LLM
    """
    is_shared = kwargs.get('is_shared', True)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 按用户流量需求降序排序处理
    sorted_users = sorted(users.items(), key=lambda x: x[1].bw, reverse=True)

    # 按流量需求降序处理用户
    for uid, user in sorted_users:
        demand = user.bw

        # 找到该用户的第一个可用LLM
        ideal_llms = user_ideal_llms.get(uid, {})
        target_llm = None

        if ideal_llms:
            # 按距离排序，取第一个可用的
            for lid, _ in sorted(ideal_llms.items(), key=lambda x: x[1]):
                if lid not in llms_copy:
                    continue
                if is_shared and llms_copy[lid].service_capacity < demand:
                    continue
                target_llm = lid
                break

        if target_llm is None:
            # 如果没有ideal_llm可用，跳过该用户
            continue

        # 计算到target_llm的路径
        dist, prev = net.dijkstra_with_capacity(uid,
                                                demand,
                                                target_id=target_llm)

        if dist[target_llm] >= float('inf'):
            continue

        path = net.get_path(prev, uid, target_llm)
        if not path:
            continue

        path_links = []
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    break

        net.send_flow(path_links, demand)

        if is_shared:
            llms_copy[target_llm].service_capacity -= demand

        path_cost = sum(link.distance for link in path_links) * demand
        allocations.append((uid, target_llm, path, demand, path_cost))
        total_cost += path_cost
        served_flow += demand

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'network': net
    }


def no_split_aggregate(network, users: Dict, llms: Dict,
                       **kwargs) -> Dict[str, Any]:
    """
    不分流算法（带汇集图，只有超汇T）

    策略：
    - 只创建超汇T，所有LLM连接到T
    - 每轮从每个用户出发路由到T，选择花销最小的路径
    - 推送用户的完整流量（不分流）
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级汇点T（没有超源S）
    T = -2
    import Entity
    net.add_node(Entity.Node(T, 0, 0))

    # 连接所有LLM到T（单向边）
    for lid, llm in llms_copy.items():
        capacity = llm.service_capacity if is_shared else float('inf')
        net.add_one_way_link(lid, T, capacity, epsilon_cost)

    allocations = []
    total_cost = 0.0
    total_demand = sum(u.bw for u in users.values())
    served_flow = 0.0

    # 记录每个用户是否已分配
    user_allocated = {uid: False for uid in users.keys()}

    while True:
        # 每轮从每个用户出发到T寻路，选择花销最小的
        best_cost = float('inf')
        best_user = None
        best_llm = None
        best_path = None
        best_path_links = None

        for uid, user in users.items():
            if user_allocated[uid]:
                continue

            demand = user.bw

            # 从该用户出发到T的最短路径
            dist, prev = net.dijkstra_with_capacity(uid, demand, target_id=T)

            if dist[T] >= float('inf'):
                continue

            # 重建路径
            path = net.get_path(prev, uid, T)
            if len(path) < 3:  # 至少需要user->...->LLM->T
                continue

            actual_llm = path[-2]  # T前面的节点是LLM

            # 验证
            if actual_llm not in llms_copy:
                continue

            # 重建路径上的边
            path_links = []
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                for link in net.links.get(src, []):
                    if link.dst == dst and not link.is_reverse:
                        path_links.append(link)
                        bottleneck = min(bottleneck, link.residual_capacity)
                        break

            # 检查瓶颈容量是否满足需求
            if bottleneck < demand - 1e-9:
                continue

            # 计算成本（排除LLM->T边）
            path_cost = sum(link.distance for link in path_links[:-1]) * demand

            # 选择开销最小的
            if path_cost < best_cost:
                best_cost = path_cost
                best_user = uid
                best_llm = actual_llm
                best_path = path
                best_path_links = path_links

        # 如果本轮没有找到可行路径，结束
        if best_user is None:
            break

        demand = users[best_user].bw

        # 推送流量
        net.send_flow(best_path_links, demand)

        # 更新LLM容量
        if is_shared:
            llms_copy[best_llm].service_capacity -= demand

        # 标记用户已分配
        user_allocated[best_user] = True

        # 记录分配（去掉T节点）
        path_without_T = best_path[:-1]
        allocations.append(
            (best_user, best_llm, path_without_T, demand, best_cost))
        total_cost += best_cost
        served_flow += demand

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'network': net
    }


def bottleneck_split(network, users: Dict, llms: Dict, user_ideal_llms: Dict,
                     **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流算法（汇集图版本）

    策略：
    - 利用汇集图结构（S→users→network→LLMs→T）
    - 每轮直接从S到T找最短路径（Dijkstra，不考虑反向边）
    - 使用瓶颈流量作为推流粒度
    - 允许用户流量分流到不同LLM
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
    search_space = 0  # 记录推流次数

    while True:
        # 直接从S到T找最短路径（Dijkstra，只考虑正向边）
        dist, prev = net.dijkstra_with_capacity(S, 1e-9, target_id=T)

        if dist[T] >= float('inf'):
            break

        # 重建路径
        path = net.get_path(prev, S, T)
        if len(path) < 4:  # 至少需要 S->user->...->LLM->T
            break

        # 提取user和LLM
        user_id = path[1]  # S后第一个节点是user
        llm_id = path[-2]  # T前最后一个节点是LLM

        # 验证
        if user_id not in users or llm_id not in llms_copy:
            break

        # 重建路径上的边，计算瓶颈容量
        path_links = []
        bottleneck = float('inf')
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for link in net.links.get(src, []):
                if link.dst == dst and not link.is_reverse:
                    path_links.append(link)
                    bottleneck = min(bottleneck, link.residual_capacity)
                    break

        if bottleneck <= 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, bottleneck)
        search_space += 1  # 每次推流计数

        # 更新LLM容量
        if is_shared:
            llms_copy[llm_id].service_capacity -= bottleneck

        # 计算成本（排除S->user和LLM->T的虚拟边）
        path_cost = sum(link.distance for link in path_links[1:-1]) * bottleneck

        # 记录分配（只记录user->...->LLM的真实路径）
        user_to_llm_path = path[1:-1]
        allocations.append(
            (user_id, llm_id, user_to_llm_path, bottleneck, path_cost))
        total_cost += path_cost
        served_flow += bottleneck

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'search_space': search_space,
        'network': net
    }


def bottleneck_split_no_aggregate(network, users: Dict, llms: Dict,
                                  user_ideal_llms: Dict,
                                  **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流算法（不带汇集图）

    策略：
    - 将user流量分流到user_ideal_llm前几个llm
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
    search_space = 0  # 记录尝试路由的次数

    # 按用户流量需求降序排序处理
    sorted_users = sorted(users.items(), key=lambda x: x[1].bw, reverse=True)

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

            # 对同一个LLM，重复寻路推流，直到无法找到新路径或需求满足
            while remaining > 1e-9:
                if is_shared and llms_copy[lid].service_capacity < 1e-9:
                    break  # LLM容量耗尽

                # 使用Dijkstra找路径（min_capacity=1，只需要找到可用路径）
                dist, prev = net.dijkstra_with_capacity(uid,
                                                        min_capacity=1,
                                                        target_id=lid)
                search_space += 1  # 每次尝试路由计数

                if dist[lid] >= float('inf'):
                    break  # 找不到可行路径，尝试下一个LLM

                path = net.get_path(prev, uid, lid)
                path_links = []
                bottleneck = float('inf')

                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    for link in net.links.get(src, []):
                        if link.dst == dst and not link.is_reverse:
                            path_links.append(link)
                            bottleneck = min(bottleneck,
                                             link.residual_capacity)
                            break

                # 考虑LLM容量限制
                if is_shared:
                    bottleneck = min(bottleneck,
                                     llms_copy[lid].service_capacity)

                # 推送瓶颈流量
                push_flow = min(bottleneck, remaining)

                if push_flow > 1e-9:
                    net.send_flow(path_links, push_flow)

                    if is_shared:
                        llms_copy[lid].service_capacity -= push_flow

                    path_cost = sum(link.distance
                                    for link in path_links) * push_flow
                    allocations.append((uid, lid, path, push_flow, path_cost))
                    total_cost += path_cost
                    served_flow += push_flow
                    remaining -= push_flow
                else:
                    break  # 推流量为0，说明路径已饱和

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'search_space': search_space,
        'network': net
    }


def one_split_no_aggregate(network, users: Dict, llms: Dict,
                           user_ideal_llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    1-split算法（不带汇集图）

    策略：
    - 将user流量分流到user_ideal_llm前几个llm
    - 每轮推流粒度固定为1

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
    search_space = 0  # 记录尝试路由的次数

    # 记录每一轮的推流状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0

    # 按用户流量需求降序排序处理
    sorted_users = sorted(users.items(), key=lambda x: x[1].bw, reverse=True)

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

            # 找一次路径（min_capacity=1，只需要找到可用路径）
            dist, prev = net.dijkstra_with_capacity(uid,
                                                    min_capacity=1,
                                                    target_id=lid)

            if dist[lid] < float('inf'):
                path = net.get_path(prev, uid, lid)
                path_links = []

                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    for link in net.links.get(src, []):
                        if link.dst == dst and not link.is_reverse:
                            path_links.append(link)
                            break

                # 在这条路径上循环推1单位，直到路径饱和或需求满足
                while remaining > 1e-9:
                    # 计算当前路径瓶颈
                    bottleneck = min(link.residual_capacity
                                     for link in path_links)

                    # 考虑LLM容量限制
                    if is_shared:
                        bottleneck = min(bottleneck,
                                         llms_copy[lid].service_capacity)

                    # 推流粒度固定为1
                    push_flow = min(1, bottleneck, remaining)

                    if push_flow < 1e-9:
                        break  # 路径饱和，尝试下一个LLM

                    net.send_flow(path_links, push_flow)
                    search_space += 1  # 每次推流计数

                    if is_shared:
                        llms_copy[lid].service_capacity -= push_flow

                    path_cost = sum(link.distance
                                    for link in path_links) * push_flow
                    allocations.append((uid, lid, path, push_flow, path_cost))
                    total_cost += path_cost
                    served_flow += push_flow
                    remaining -= push_flow

                    # 更新累积数据
                    cumulative_cost += path_cost
                    cumulative_flow += push_flow
                    round_num += 1

                    # 记录本轮状态
                    round_allocations.append({
                        'round': round_num,
                        'push_flow': push_flow,
                        'round_cost': path_cost,
                        'cumulative_cost': cumulative_cost,
                        'cumulative_flow': cumulative_flow,
                        'path': path
                    })

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'search_space': search_space,
        'network': net
    }


def LLM_split(network, users: Dict, llms: Dict, user_ideal_llms: Dict,
              **kwargs) -> Dict[str, Any]:
    """
    LLM分流算法

    策略：
    - 将user流量分流到user_ideal_llm前几个llm
    - 分流粒度为LLM剩余容量

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

    # 按用户流量需求降序排序处理
    sorted_users = sorted(users.items(), key=lambda x: x[1].bw, reverse=True)

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

            llm_capacity = llms_copy[
                lid].service_capacity if is_shared else float('inf')
            if llm_capacity < 1e-9:
                continue

            # 分流粒度为LLM剩余容量
            push_flow = min(llm_capacity, remaining)

            dist, prev = net.dijkstra_with_capacity(uid,
                                                    push_flow,
                                                    target_id=lid)

            if dist[lid] < float('inf'):
                path = net.get_path(prev, uid, lid)
                path_links = []
                bottleneck = float('inf')

                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    for link in net.links.get(src, []):
                        if link.dst == dst and not link.is_reverse:
                            path_links.append(link)
                            bottleneck = min(bottleneck,
                                             link.residual_capacity)
                            break

                # 实际推送流量受路径瓶颈限制
                actual_push = min(push_flow, bottleneck)

                if actual_push > 1e-9:
                    net.send_flow(path_links, actual_push)

                    if is_shared:
                        llms_copy[lid].service_capacity -= actual_push

                    path_cost = sum(link.distance
                                    for link in path_links) * actual_push
                    allocations.append(
                        (uid, lid, path, actual_push, path_cost))
                    total_cost += path_cost
                    served_flow += actual_push
                    remaining -= actual_push

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'network': net
    }


def LLM_split_aggregate(network, users: Dict, llms: Dict,
                        user_ideal_llms: Dict, **kwargs) -> Dict[str, Any]:
    """
    LLM分流算法（带汇集图，只有超汇T）

    策略：
    - 只创建超汇T，所有LLM连接到T
    - 每轮从每个用户出发路由到T，选择花销最小的路径
    - 推流粒度为min(用户剩余需求, 最小LLM剩余容量)
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # 创建超级汇点T（没有超源S）
    T = -2
    import Entity
    net.add_node(Entity.Node(T, 0, 0))

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
        # 计算最小LLM剩余容量
        if is_shared:
            min_llm = min(
                (llm.service_capacity
                 for llm in llms_copy.values() if llm.service_capacity > 1e-9),
                default=0.0)
        else:
            min_llm = float('inf')

        if min_llm <= 1e-9:
            break

        # 每轮从每个用户出发到T寻路，选择花销最小的
        best_cost = float('inf')
        best_user = None
        best_llm = None
        best_path = None
        best_path_links = None
        best_push_flow = 0

        for uid, user in users.items():
            if user_remaining[uid] <= 1e-9:
                continue

            # 推流粒度为min(用户剩余需求, 最小LLM剩余容量)
            push_flow = min(user_remaining[uid], min_llm)

            # 从该用户出发到T的最短路径
            dist, prev = net.dijkstra_with_capacity(uid,
                                                    push_flow,
                                                    target_id=T)

            if dist[T] >= float('inf'):
                continue

            # 重建路径
            path = net.get_path(prev, uid, T)
            if len(path) < 3:  # 至少需要user->...->LLM->T
                continue

            actual_llm = path[-2]  # T前面的节点是LLM

            # 验证
            if actual_llm not in llms_copy:
                continue

            # 重建路径上的边
            path_links = []
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                for link in net.links.get(src, []):
                    if link.dst == dst and not link.is_reverse:
                        path_links.append(link)
                        bottleneck = min(bottleneck, link.residual_capacity)
                        break

            # 检查瓶颈容量
            if bottleneck < push_flow - 1e-9:
                continue

            # 计算成本（排除LLM->T边）
            path_cost = sum(link.distance
                            for link in path_links[:-1]) * push_flow

            # 选择开销最小的
            if path_cost < best_cost:
                best_cost = path_cost
                best_user = uid
                best_llm = actual_llm
                best_path = path
                best_path_links = path_links
                best_push_flow = push_flow

        # 如果本轮没有找到可行路径，结束
        if best_user is None:
            break

        # 推送流量
        net.send_flow(best_path_links, best_push_flow)

        # 更新用户剩余流量
        user_remaining[best_user] -= best_push_flow

        # 更新LLM容量
        if is_shared:
            llms_copy[best_llm].service_capacity -= best_push_flow

        # 记录分配（去掉T节点）
        path_without_T = best_path[:-1]
        allocations.append(
            (best_user, best_llm, path_without_T, best_push_flow, best_cost))
        total_cost += best_cost
        served_flow += best_push_flow

    acceptance_ratio = (served_flow /
                        total_demand if total_demand > 0 else 0.0)

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'network': net
    }


def bottleneck_split_augment(network, users: Dict, llms: Dict,
                             **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流增广算法（带汇集图+负环消除）
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

    # 记录每一轮的推流状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0
    search_space = 0  # 记录SPFA调用次数

    # 定义 SPFA 查找增广路径（在残量网络上，含反向边）
    def find_augmenting_path_spfa(source, sink):
        """
        使用 SPFA 在残量网络上查找最短增广路径

        返回:
            (dist, prev_with_links) 其中 prev_with_links[v] = (u, link)
        """
        from collections import deque

        dist = {nid: float('inf') for nid in net.nodes}
        prev = {nid: None for nid in net.nodes}
        in_queue = {nid: False for nid in net.nodes}

        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            for link in net.links.get(u, []):
                # 检查残余容量
                if link.residual_capacity < 1e-9:
                    continue

                v = link.dst
                # 计算成本：反向边使用负成本
                cost = link.distance if not link.is_reverse else -link.reverse.distance

                if dist[u] + cost < dist[v] - 1e-9:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, link)  # 记录边对象！

                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        return dist, prev

    # 循环增广
    while True:
        # 消除负环
        net.cancel_negative_cycles()

        # 使用 SPFA 查找增广路径（含反向边）
        dist, prev = find_augmenting_path_spfa(S, T)

        if dist[T] >= float('inf'):
            break

        # 提取路径：使用记录的边对象
        path_links = []
        path_nodes = []
        cur = T
        while cur != S:
            if prev[cur] is None:
                break  # 路径不完整
            prev_node, prev_link = prev[cur]
            path_links.append(prev_link)
            path_nodes.append(cur)
            cur = prev_node

        path_nodes.append(S)
        path_nodes.reverse()
        path_links.reverse()

        # 计算瓶颈容量
        bottleneck = min(link.residual_capacity for link in path_links)

        if bottleneck < 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, bottleneck)
        search_space += 1  # 每次推流计数

        # 计算本轮实际成本（排除 S->user 和 LLM->T 的虚拟边）
        round_cost = 0.0
        for i, link in enumerate(path_links):
            if i == 0 or i == len(path_links) - 1:  # 跳过第一条和最后一条
                continue
            # 正向边加成本，反向边减成本（因为反向边是回收流量）
            if not link.is_reverse:
                round_cost += link.distance * bottleneck
            else:
                round_cost -= link.reverse.distance * bottleneck

        cumulative_cost += round_cost
        cumulative_flow += bottleneck
        round_num += 1

        # 记录本轮状态
        round_allocations.append({
            'round': round_num,
            'push_flow': bottleneck,
            'round_cost': round_cost,
            'cumulative_cost': cumulative_cost,
            'cumulative_flow': cumulative_flow,
            'path': path_nodes[1:-1]  # 去掉 S 和 T
        })

    # 最终用 BFS 重建一次，得到完整的分配结果
    allocations = []
    for uid, user in users.items():
        user_allocations = _trace_user_flow_bfs(net, uid, users, llms_copy, S,
                                                T)
        allocations.extend(user_allocations)

    # 计算最终总成本和服务流量
    total_cost = cumulative_cost
    served_flow = cumulative_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'search_space': search_space,
        'network': net
    }


def one_split(network, users: Dict, llms: Dict, user_ideal_llms: Dict,
              **kwargs) -> Dict[str, Any]:
    """
    1-split算法（汇集图版本）

    策略：
    - 利用汇集图结构（S→users→network→LLMs→T）
    - 每轮直接从S到T找最短路径（Dijkstra，不考虑反向边）
    - 推流粒度固定为1
    - 允许用户流量分流到不同LLM
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
    search_space = 0  # 记录Dijkstra调用次数

    # 记录每一轮的推流状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0

    while True:
        # 直接从S到T找最短路径（Dijkstra，只考虑正向边）
        dist, prev = net.dijkstra_with_capacity(S, 1, target_id=T)

        if dist[T] >= float('inf'):
            break

        # 重建路径
        path = net.get_path(prev, S, T)
        if len(path) < 4:  # 至少需要 S->user->...->LLM->T
            break

        # 提取user和LLM
        user_id = path[1]  # S后第一个节点是user
        llm_id = path[-2]  # T前最后一个节点是LLM

        # 验证
        if user_id not in users or llm_id not in llms_copy:
            break

        # 重建路径上的边，计算瓶颈容量
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
        if push_flow <= 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, push_flow)
        search_space += 1  # 每次推流计数

        # 更新LLM容量
        if is_shared:
            llms_copy[llm_id].service_capacity -= push_flow

        # 计算成本（排除S->user和LLM->T的虚拟边）
        path_cost = sum(link.distance for link in path_links[1:-1]) * push_flow

        # 记录分配（只记录user->...->LLM的真实路径）
        user_to_llm_path = path[1:-1]
        allocations.append(
            (user_id, llm_id, user_to_llm_path, push_flow, path_cost))
        total_cost += path_cost
        served_flow += push_flow

        # 更新累积数据
        cumulative_cost += path_cost
        cumulative_flow += push_flow
        round_num += 1

        # 记录本轮状态
        round_allocations.append({
            'round': round_num,
            'push_flow': push_flow,
            'round_cost': path_cost,
            'cumulative_cost': cumulative_cost,
            'cumulative_flow': cumulative_flow,
            'path': user_to_llm_path
        })

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'search_space': search_space,
        'network': net
    }


def one_split_augment(network, users: Dict, llms: Dict,
                      **kwargs) -> Dict[str, Any]:
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

    # 记录每一轮的推流状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0
    search_space = 0  # 记录SPFA调用次数

    # 定义 SPFA 查找增广路径（在残量网络上，含反向边）
    def find_augmenting_path_spfa(source, sink):
        """
        使用 SPFA 在残量网络上查找最短增广路径

        返回:
            (dist, prev_with_links) 其中 prev_with_links[v] = (u, link)
        """
        from collections import deque

        dist = {nid: float('inf') for nid in net.nodes}
        prev = {nid: None for nid in net.nodes}
        in_queue = {nid: False for nid in net.nodes}

        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            for link in net.links.get(u, []):
                # 检查残余容量
                if link.residual_capacity < 1e-9:
                    continue

                v = link.dst
                # 计算成本：反向边使用负成本
                cost = link.distance if not link.is_reverse else -link.reverse.distance

                if dist[u] + cost < dist[v] - 1e-9:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, link)  # 记录边对象！

                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        return dist, prev

    # 循环增广，每次推流粒度为1
    while True:
        # 消除负环
        net.cancel_negative_cycles()

        # 使用 SPFA 查找增广路径（含反向边）
        dist, prev = find_augmenting_path_spfa(S, T)

        if dist[T] >= float('inf'):
            break

        # 提取路径：使用记录的边对象（参考 successive_shortest_paths）
        path_links = []
        path_nodes = []
        cur = T
        while cur != S:
            if prev[cur] is None:
                break  # 路径不完整
            prev_node, prev_link = prev[cur]
            path_links.append(prev_link)
            path_nodes.append(cur)
            cur = prev_node

        path_nodes.append(S)
        path_nodes.reverse()
        path_links.reverse()

        # 计算瓶颈容量
        bottleneck = min(link.residual_capacity for link in path_links)

        # 推流粒度为1
        push_flow = min(1, bottleneck)

        if push_flow < 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, push_flow)
        search_space += 1  # 每次推流计数

        # 计算本轮实际成本（排除 S->user 和 LLM->T 的虚拟边）
        round_cost = 0.0
        for i, link in enumerate(path_links):
            if i == 0 or i == len(path_links) - 1:  # 跳过第一条和最后一条
                continue
            # 正向边加成本，反向边减成本（因为反向边是回收流量）
            if not link.is_reverse:
                round_cost += link.distance * push_flow
            else:
                round_cost -= link.reverse.distance * push_flow

        cumulative_cost += round_cost
        cumulative_flow += push_flow
        round_num += 1

        # 记录本轮状态
        round_allocations.append({
            'round': round_num,
            'push_flow': push_flow,
            'round_cost': round_cost,
            'cumulative_cost': cumulative_cost,
            'cumulative_flow': cumulative_flow,
            'path': path_nodes[1:-1]  # 去掉 S 和 T
        })

    # 最终用 BFS 重建一次，得到完整的分配结果
    allocations = []
    for uid, user in users.items():
        user_allocations = _trace_user_flow_bfs(net, uid, users, llms_copy, S,
                                                T)
        allocations.extend(user_allocations)

    # 计算最终总成本和服务流量
    total_cost = cumulative_cost
    served_flow = cumulative_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'search_space': search_space,
        'network': net
    }


def NW_bottleneck_split_augment(network, users: Dict, llms: Dict,
                                  **kwargs) -> Dict[str, Any]:
    """
    瓶颈分流增广算法（不允许LLM反向边版本）

    NW = No Withdraw
    策略：与bottleneck_split_augment相同，但删除LLM的所有反向残余边
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # NW版本：删除LLM节点的所有边（都是反向残余边）
    for llm_id in llms_copy.keys():
        if llm_id in net.links:
            net.links[llm_id] = []

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

    # 记录每一轮的推流状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0

    # 定义 SPFA 查找增广路径（在残量网络上，含反向边）
    def find_augmenting_path_spfa(source, sink):
        """
        使用 SPFA 在残量网络上查找最短增广路径

        返回:
            (dist, prev_with_links) 其中 prev_with_links[v] = (u, link)
        """
        from collections import deque

        dist = {nid: float('inf') for nid in net.nodes}
        prev = {nid: None for nid in net.nodes}
        in_queue = {nid: False for nid in net.nodes}

        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            for link in net.links.get(u, []):
                # 检查残余容量
                if link.residual_capacity < 1e-9:
                    continue

                v = link.dst
                # 计算成本：反向边使用负成本
                cost = link.distance if not link.is_reverse else -link.reverse.distance

                if dist[u] + cost < dist[v] - 1e-9:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, link)  # 记录边对象！

                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        return dist, prev

    # 循环增广
    while True:
        # 消除负环
        net.cancel_negative_cycles()

        # 使用 SPFA 查找增广路径（含反向边）
        dist, prev = find_augmenting_path_spfa(S, T)

        if dist[T] >= float('inf'):
            break

        # 提取路径：使用记录的边对象
        path_links = []
        path_nodes = []
        cur = T
        while cur != S:
            if prev[cur] is None:
                break  # 路径不完整
            prev_node, prev_link = prev[cur]
            path_links.append(prev_link)
            path_nodes.append(cur)
            cur = prev_node

        path_nodes.append(S)
        path_nodes.reverse()
        path_links.reverse()

        # 计算瓶颈容量
        bottleneck = min(link.residual_capacity for link in path_links)

        if bottleneck < 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, bottleneck)

        # 计算本轮实际成本（排除 S->user 和 LLM->T 的虚拟边）
        round_cost = 0.0
        for i, link in enumerate(path_links):
            if i == 0 or i == len(path_links) - 1:  # 跳过第一条和最后一条
                continue
            # 正向边加成本，反向边减成本（因为反向边是回收流量）
            if not link.is_reverse:
                round_cost += link.distance * bottleneck
            else:
                round_cost -= link.reverse.distance * bottleneck

        cumulative_cost += round_cost
        cumulative_flow += bottleneck
        round_num += 1

        # 记录本轮状态（包含 user_id 和 llm_id）
        path_without_st = path_nodes[1:-1]  # 去掉 S 和 T
        user_id = path_without_st[0] if path_without_st else None
        llm_id = path_without_st[-1] if path_without_st else None

        round_allocations.append({
            'round': round_num,
            'push_flow': bottleneck,
            'round_cost': round_cost,
            'cumulative_cost': cumulative_cost,
            'cumulative_flow': cumulative_flow,
            'path': path_without_st,
            'user_id': user_id,
            'llm_id': llm_id
        })

    # 直接从 round_allocations 生成 allocations（不使用 _trace_user_flow_bfs）
    allocations = []
    for ra in round_allocations:
        if ra['user_id'] is not None and ra['llm_id'] is not None:
            allocations.append((
                ra['user_id'],
                ra['llm_id'],
                ra['path'],
                ra['push_flow'],
                ra['round_cost']
            ))

    # 计算最终总成本和服务流量
    total_cost = cumulative_cost
    served_flow = cumulative_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'network': net
    }


def NW_one_split_augment(network, users: Dict, llms: Dict,
                          **kwargs) -> Dict[str, Any]:
    """
    1-split增广算法（不允许LLM反向边版本）

    NW = No Withdraw
    策略：与one_split_augment相同，但删除LLM的所有反向残余边
    """
    is_shared = kwargs.get('is_shared', True)
    epsilon_cost = kwargs.get('epsilon_cost', 1e-9)

    net = copy.deepcopy(network)
    llms_copy = copy.deepcopy(llms)

    # NW版本：删除LLM节点的所有边（都是反向残余边）
    for llm_id in llms_copy.keys():
        if llm_id in net.links:
            net.links[llm_id] = []

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

    # 记录每一轮的推流状态
    round_allocations = []
    cumulative_cost = 0.0
    cumulative_flow = 0.0
    round_num = 0

    # 定义 SPFA 查找增广路径（在残量网络上，含反向边）
    def find_augmenting_path_spfa(source, sink):
        """
        使用 SPFA 在残量网络上查找最短增广路径

        返回:
            (dist, prev_with_links) 其中 prev_with_links[v] = (u, link)
        """
        from collections import deque

        dist = {nid: float('inf') for nid in net.nodes}
        prev = {nid: None for nid in net.nodes}
        in_queue = {nid: False for nid in net.nodes}

        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            for link in net.links.get(u, []):
                # 检查残余容量
                if link.residual_capacity < 1e-9:
                    continue

                v = link.dst
                # 计算成本：反向边使用负成本
                cost = link.distance if not link.is_reverse else -link.reverse.distance

                if dist[u] + cost < dist[v] - 1e-9:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, link)  # 记录边对象！

                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        return dist, prev

    # 循环增广，每次推流粒度为1
    while True:
        # 消除负环
        net.cancel_negative_cycles()

        # 使用 SPFA 查找增广路径（含反向边）
        dist, prev = find_augmenting_path_spfa(S, T)

        if dist[T] >= float('inf'):
            break

        # 提取路径：使用记录的边对象
        path_links = []
        path_nodes = []
        cur = T
        while cur != S:
            if prev[cur] is None:
                break  # 路径不完整
            prev_node, prev_link = prev[cur]
            path_links.append(prev_link)
            path_nodes.append(cur)
            cur = prev_node

        path_nodes.append(S)
        path_nodes.reverse()
        path_links.reverse()

        # 计算瓶颈容量
        bottleneck = min(link.residual_capacity for link in path_links)

        # 推流粒度为1
        push_flow = min(1, bottleneck)

        if push_flow < 1e-9:
            break

        # 推送流量
        net.send_flow(path_links, push_flow)

        # 计算本轮实际成本（排除 S->user 和 LLM->T 的虚拟边）
        round_cost = 0.0
        for i, link in enumerate(path_links):
            if i == 0 or i == len(path_links) - 1:  # 跳过第一条和最后一条
                continue
            # 正向边加成本，反向边减成本（因为反向边是回收流量）
            if not link.is_reverse:
                round_cost += link.distance * push_flow
            else:
                round_cost -= link.reverse.distance * push_flow

        cumulative_cost += round_cost
        cumulative_flow += push_flow
        round_num += 1

        # 记录本轮状态（包含 user_id 和 llm_id）
        path_without_st = path_nodes[1:-1]  # 去掉 S 和 T
        user_id = path_without_st[0] if path_without_st else None
        llm_id = path_without_st[-1] if path_without_st else None

        round_allocations.append({
            'round': round_num,
            'push_flow': push_flow,
            'round_cost': round_cost,
            'cumulative_cost': cumulative_cost,
            'cumulative_flow': cumulative_flow,
            'path': path_without_st,
            'user_id': user_id,
            'llm_id': llm_id
        })

    # 直接从 round_allocations 生成 allocations（不使用 _trace_user_flow_bfs）
    allocations = []
    for ra in round_allocations:
        if ra['user_id'] is not None and ra['llm_id'] is not None:
            allocations.append((
                ra['user_id'],
                ra['llm_id'],
                ra['path'],
                ra['push_flow'],
                ra['round_cost']
            ))

    # 计算最终总成本和服务流量
    total_cost = cumulative_cost
    served_flow = cumulative_flow

    acceptance_ratio = served_flow / total_demand if total_demand > 0 else 0.0

    return {
        'allocations': allocations,
        'total_cost': total_cost,
        'round_allocations': round_allocations,
        'total_flow': total_demand,
        'served_flow': served_flow,
        'acceptance_ratio': acceptance_ratio,
        'network': net
    }


# 算法注册表
ALGORITHMS = {
    'no-split': no_split,
    'no-split-aggregate': no_split_aggregate,
    'bottleneck-split': bottleneck_split,
    'bottleneck-split-no-aggregate': bottleneck_split_no_aggregate,
    'LLM-split': LLM_split,
    'LLM-split-aggregate': LLM_split_aggregate,
    'bottleneck-split-augment': bottleneck_split_augment,
    '1-split': one_split,
    '1-split-no-aggregate': one_split_no_aggregate,
    '1-split-augment': one_split_augment,
    'NW-bottleneck-split-augment': NW_bottleneck_split_augment,
    'NW-1-split-augment': NW_one_split_augment,
}


def run_algorithm(algorithm_name: str, network, users: Dict, llms: Dict,
                  **kwargs) -> Dict[str, Any]:
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
        raise ValueError(
            f"未知算法: {algorithm_name}. 可用算法: {list(ALGORITHMS.keys())}")

    return ALGORITHMS[algorithm_name](network, users, llms, **kwargs)
