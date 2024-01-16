def check_null(graph_1):
    for node_1, node_data_1 in graph_1.nodes(data=True):
        if(node_data_1 == {}):
            for source, target, edge_data in graph_1.edges(data=True):
                if(source == node_1 or target == node_1):
                    print(node_1, ':', source, target, edge_data)
                    return False
    else:
        return True

def have_same(graph_1, graph_2):
    for node_1 in graph_1.nodes():
        for node_2 in graph_2.nodes():
            if node_1 == node_2:
                return True
    return False