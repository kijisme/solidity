import os
import networkx as nx

from cfg import get_vuln_cfg_graph
from cg import get_vuln_call_graph
from combineGraph import *

def get_all_graph(vuln_dataset_dir, isSave = False):

    get_vuln_cfg_graph(vuln_dataset_dir, isSave)
    get_vuln_call_graph(vuln_dataset_dir, isSave)

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


if __name__ == "__main__":

    vuln_dataset_dir = '/workspaces/solidity/integrate/1'
    isSave = True
    get_all_graph(vuln_dataset_dir, isSave)