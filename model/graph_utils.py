
import networkx as nx

import torch

def add_hetero_ids(nx_graph):
    nx_g = nx_graph
    dict_hetero_id = {}

    for node, node_data in nx_g.nodes(data=True):
        if node_data['node_type'] not in dict_hetero_id:
            dict_hetero_id[node_data['node_type']] = 0
        else:
            dict_hetero_id[node_data['node_type']] += 1
        nx_g.nodes[node]['node_hetero_id'] = dict_hetero_id[node_data['node_type']]
    return nx_g


def load_hetero_nx_graph(nx_graph_path):
    # 读取图
    nx_graph = nx.read_gpickle(nx_graph_path)
    # 将图标签转化为数字
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    # 添加节点索引 'node_hetero_id'
    nx_graph = add_hetero_ids(nx_graph)
    return nx_graph

def convert_edge_data_to_tensor(dict_egdes):
    dict_three_cannonical_egdes = dict_egdes
    for key, val in dict_three_cannonical_egdes.items():
        list_source = []
        list_target = []
        for source, target in val:
            list_source.append(source)
            list_target.append(target)
        dict_three_cannonical_egdes[key] = (torch.tensor(list_source), torch.tensor(list_target))
    return dict_three_cannonical_egdes

def generate_hetero_graph_data(nx_graph):
    nx_g = nx_graph
    # 获取三元组(source_type, edge_type, target_type) -> [(source_id, target_id)]
    dict_three_cannonical_egdes = dict()
    for source, target, data in nx_g.edges(data=True):
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        three_cannonical_egde = (source_node_type, edge_type, target_node_type)

        if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
            dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])]
        else:
            current_val = dict_three_cannonical_egdes[three_cannonical_egde]
            temp_edge = (nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])
            current_val.append(temp_edge)
            dict_three_cannonical_egdes[three_cannonical_egde] = current_val
    # 将列表值[(source_id, target_id)]  转化为 ([all_source],[all_target])
    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)
    return dict_three_cannonical_egdes

def generate_filename_ids(nx_graph):
    file_ids = {}
    for _, node_data in nx_graph.nodes(data=True):
        filename = node_data['source_file']
        if filename not in file_ids:
            file_ids[filename] = len(file_ids)
    return file_ids

def get_node_tracker(nx_graph, filename_mapping):
    nx_g = nx_graph
    node_tracker = {}
    for _, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        filename = node_data['source_file']
        if node_type not in node_tracker.keys():
            node_tracker[node_type] = torch.tensor([filename_mapping[filename]], dtype=torch.int64)
        else:
            node_tracker[node_type] = torch.cat((node_tracker[node_type], torch.tensor([filename_mapping[filename]], dtype=torch.int64)))
    return node_tracker

def reflect_graph(nx_g_data):
    symmetrical_data = {}
    for metapath, value in nx_g_data.items():
        # source_type和target_type相同
        if metapath[0] == metapath[-1]:
            symmetrical_data[metapath] = (torch.cat((value[0], value[1])), torch.cat((value[1], value[0])))
        else:
            # 添加正向source -> target
            if metapath not in symmetrical_data.keys():
                symmetrical_data[metapath] = value
            else:
                symmetrical_data[metapath] = (torch.cat((symmetrical_data[metapath][0], value[0])), torch.cat((symmetrical_data[metapath][1], value[1])))
            # 添加反向target -> source
            if metapath[::-1] not in symmetrical_data.keys():
                symmetrical_data[metapath[::-1]] = (value[1], value[0])
            else:
                symmetrical_data[metapath[::-1]] = (torch.cat((symmetrical_data[metapath[::-1]][0], value[1])), torch.cat((symmetrical_data[metapath[::-1]][1], value[0])))
    return symmetrical_data

# 获取每种类型节点的数目
def get_number_of_nodes(nx_graph):
    nx_g = nx_graph
    number_of_nodes = {}
    for _, data in nx_g.nodes(data=True):
        if data['node_type'] not in number_of_nodes.keys():
            number_of_nodes[data['node_type']] = 1
        else:
            number_of_nodes[data['node_type']] += 1
    return number_of_nodes

def get_length_2_metapath(symmetrical_global_graph):
    # 开始节点和结束节点序列
    begin_by = {}
    end_by = {}
    for mt in symmetrical_global_graph.canonical_etypes:
        if mt[0] not in begin_by:
            begin_by[mt[0]] = [mt]
        else:
            begin_by[mt[0]].append(mt)
        if mt[-1] not in end_by:
            end_by[mt[-1]] = [mt]
        else:
            end_by[mt[-1]].append(mt)
    
    # len_2元路径序列[([source == target]), ([ first + (first_target == second_source)])]
    metapath_list = []
    for mt_0 in symmetrical_global_graph.canonical_etypes:
        source = mt_0[0]
        dest = mt_0[-1]
        if source == dest:
            metapath_list.append([mt_0])
        first_metapath = [mt_0]
        if dest in begin_by:
            for mt_1 in begin_by[dest]:
                if mt_1 != mt_0 and mt_1[-1] == source:
                    second_metapath = first_metapath + [mt_1]
                    metapath_list.append(second_metapath)

    return metapath_list

def get_node_expression(nx_graph):
    nx_g = nx_graph
    node_expressions = {}
    for node, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        node_expression = node_data['node_expression']
        if node_type not in node_expressions.keys():
            node_expressions[node_type] = [node_expression]
        else:
            node_expressions[node_type].append(node_expression)
    
    return node_expressions