import os
import networkx as nx
from copy import deepcopy

from cfg import get_vuln_cfg_graph
from cg import get_vuln_call_graph
from combineGraph import *

def get_all_graph(dataset_dir, vuln_dataset_dir, isSave = False):

    # get_vuln_cfg_graph(dataset_dir, vuln_dataset_dir, isSave)
    get_vuln_call_graph(dataset_dir, vuln_dataset_dir, isSave)

    cfg_path = os.path.join(vuln_dataset_dir, 'cfg.gpickle')
    cg_path = os.path.join(vuln_dataset_dir, 'cg.gpickle')
    output_path = os.path.join(vuln_dataset_dir, 'compress.gpickle')
    cfg = nx.read_gpickle(cfg_path)
    cg = nx.read_gpickle(cg_path)

    # 映射图节点
    dict_node_token_cfg_and_cg = mapping_cfg_and_cg_node_token(cfg, cg)
    # 添加cfg函数调用边
    merged_graph = add_new_cfg_edges_from_call_graph(cfg, dict_node_token_cfg_and_cg, cg)
    # 更新节点类型
    update_cfg_node_types_by_call_graph_node_types(merged_graph, dict_node_token_cfg_and_cg)

    ###########################################
    from utils import check_null
    if not check_null(merged_graph):
        print('有空结点')
    else:
        print('无空结点')
    ##########################################
    # 存储图
    if isSave:
        nx.write_gpickle(merged_graph, output_path)


def concat_all_vuln(vuln_dataset_dir, ratio, target_dir):
    all_cfg = None
    all_cg = None
    vulns = [os.path.join(vuln_dataset_dir, x) for x in os.listdir(vuln_dataset_dir) if x != 'clean']
    for vuln in vulns:
        cfg_graph = os.path.join(vuln, 'integrate', str(ratio), 'cfg.gpickle')
        cg_graph = os.path.join(vuln, 'integrate', str(ratio), 'cg.gpickle')

        cfg = nx.read_gpickle(cfg_graph)
        if all_cfg is None:
            all_cfg = deepcopy(cfg)
        else:
            all_cfg = nx.disjoint_union(all_cfg, cfg)
        
        cg = nx.read_gpickle(cg_graph)
        if all_cg is None:
            all_cg = deepcopy(cg)
        else:
            all_cg = nx.disjoint_union(all_cg, cg)

    # 映射图节点
    dict_node_token_cfg_and_cg = mapping_cfg_and_cg_node_token(all_cfg, all_cg)
    # 添加cfg函数调用边
    merged_graph = add_new_cfg_edges_from_call_graph(all_cfg, dict_node_token_cfg_and_cg, all_cg)
    # 更新节点类型
    update_cfg_node_types_by_call_graph_node_types(merged_graph, dict_node_token_cfg_and_cg)
    
    # 检测非空
    # ##########################################
    from utils import check_null
    if not check_null(cfg):
        print('cfg有空结点')
    else:
        print('cfg无空结点')
    if not check_null(cg):
        print('cg有空结点')
    else:
        print('cg无空结点')
    if not check_null(merged_graph):
        print('merged_graph有空结点')
    else:
        print('merged_graph无空结点')
    ##########################################

    target_cfg = os.path.join(target_dir, str(ratio), 'cfg.gpickle')    
    target_cg = os.path.join(target_dir, str(ratio), 'cg.gpickle')    
    target_compress = os.path.join(target_dir, str(ratio), 'compress.gpickle')  

    nx.write_gpickle(all_cg, target_cg)
    nx.write_gpickle(all_cfg, target_cfg)
    nx.write_gpickle(merged_graph, target_compress)

if __name__ == "__main__":

    root_path = '/workspaces/solidity'
    
    vuln_dataset_dir = os.path.join(root_path, 'integrate_dataset')
    ratio = 1
    target_dir = os.path.join(root_path, 'integrate')

    concat_all_vuln(vuln_dataset_dir, ratio, target_dir)
    # vuln_dataset_dir = '/workspaces/solidity/integrate/1'
    # isSave = True
    # get_all_graph(dataset_dir, vuln_dataset_dir, isSave)