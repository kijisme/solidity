import os
import json
import networkx as nx

'''检查是否有空结点'''
def check_null(graph):
    for node, node_data in graph.nodes(data=True):
        if(node_data == {}):
            print('node', node)
            return False
    else:
        return True

def check_all_graph(graph_path_dir):
    for vuln in graph_path_dir:
        for graph_name in graph_path_dir[vuln]:
            graph = nx.read_gpickle(graph_path_dir[vuln][graph_name])
            if not check_null(graph):
                print(f'{vuln}_{graph_name}存在空节点')

def get_function_dir(graph):
    function_dir = {}
    for node, node_data in graph.nodes(data=True):
        if node_data['source_file'] not in function_dir.keys():
            function_dir[node_data['source_file']] = []
        if node_data['node_type'].split('_')[-1] == 'function' or node_data['node_type'] == 'FUNCTION':
            function_dir[node_data['source_file']].append(node_data['node_token'])
    return function_dir

def compare_function(cg_dir, cfg_dir):
    
    flag = True
    error_file = set()
    if cg_dir.keys() != cfg_dir.keys():
        print('文件未解析完全')
        return False
    for file in cg_dir:
        for function in cg_dir[file]:
            if function not in cfg_dir[file]:
                print(function)
                error_file.add(function.split('.sol')[0])
                flag = False
    return flag, list(error_file)
        

def check_function_in_graph(graph_path_dir):
    all_error = {}
    for vuln in graph_path_dir:
        cg = nx.read_gpickle(graph_path_dir[vuln]['cg'])
        cfg = nx.read_gpickle(graph_path_dir[vuln]['cfg'])
        cg_dir = get_function_dir(cg)
        cfg_dir = get_function_dir(cfg)
        flag, error = compare_function(cg_dir, cfg_dir)
        if not flag:
            all_error['vuln'] = error
            with open(f'/workspaces/solidity/error/{vuln}_error.json', 'w') as f:
                json.dump(all_error, f)
            print(f'{vuln}有问题')

def get_graph_dir(dataset_root):

    graph_path_dir = {}
    vuln_all = [x for x in os.listdir(dataset_root) if x != 'clean']
    for vuln in vuln_all:
        cg_path = os.path.join(dataset_root, vuln, 'integrate/cg.gpickle')
        cfg_path = os.path.join(dataset_root, vuln, 'integrate/cfg.gpickle')
        graph_path_dir[vuln] = {}
        graph_path_dir[vuln]['cg'] = cg_path
        graph_path_dir[vuln]['cfg'] = cfg_path
    return graph_path_dir

if __name__ == "__main__":
    dataset_root = '/workspaces/solidity/integrate_dataset'
    graph_path_dir = get_graph_dir(dataset_root)
    check_all_graph(graph_path_dir)
    check_function_in_graph(graph_path_dir)

    # path = '/workspaces/solidity/integrate_dataset/other/integrate/cfg.gpickle'
    # graph = nx.read_gpickle(path)
    # if not check_null(graph):
    #     print('有空结点')