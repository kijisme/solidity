def check_null(graph):
    flag = True
    nodes = []
    for node, node_data in graph.nodes(data=True):
        if(node_data == {}):
            nodes.append(node)
            flag=False
    edges = set()
    for source, target, edge_data in graph.edges(data=True):
        if source in nodes:
            edges.add(f'{source}_{graph.nodes(data=False)[target]}')
        if target in nodes:
            edges.add(f'{target}_{graph.nodes(data=False)[source]}')
    for edge in edges:
        print(edge)
    
    return flag

def have_same(graph_1, graph_2):
    for node_1 in graph_1.nodes():
        for node_2 in graph_2.nodes():
            if node_1 == node_2:
                return True
    return False