def check_null(graph):
    for node, node_data in graph.nodes(data=True):
        if(node_data == {}):
            return False
    else:
        return True

def have_same(graph_1, graph_2):
    for node_1 in graph_1.nodes():
        for node_2 in graph_2.nodes():
            if node_1 == node_2:
                return True
    return False