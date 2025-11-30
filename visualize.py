import Entity

DISTRIBUTION_TYPES = Entity.DISTRIBUTION_TYPES

for user_distribution in DISTRIBUTION_TYPES:
    for llm_distribution in DISTRIBUTION_TYPES:
        # 先加载LLM信息，然后加载网络（带LLM标识）
        llms = Entity.load_llm_info(user_distribution, llm_distribution)
        json = Entity.load_network_from_sheets(llm_ids=llms.keys())
        network = json['network']
        nodes_list = list(json['nodes'].values())
        nodes = json['nodes']
        users = Entity.load_user_info(user_distribution)
        for llm in llms.values():
            nodes_list[llm.id].role = 'llm'
            nodes_list[llm.id].deployed = 1
        for user in users.values():
            nodes_list[user.id].role = 'user'

        # 用户按带宽排序
        users = dict(
            sorted(users.items(), key=lambda item: item[1].bw, reverse=True))

        user_ideal_llms = {}
        for user in users.values():
            distances, costs = network.dijkstra_ideal(user.id, user.bw)
            sorted_nodes = sorted(distances, key=distances.get)
            ideal_llms = {
                n: costs[n]
                for n in sorted_nodes if nodes_list[n].role == 'llm'
            }
            user_ideal_llms[user.id] = ideal_llms

        Entity.visualize_network(nodes_list, network, llms, users,
                                 user_distribution, llm_distribution)
