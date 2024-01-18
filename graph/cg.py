import os
import json
import argparse	
import subprocess
import networkx as nx
from copy import deepcopy
from slither.slither import Slither

from vulnInfo import get_vulnerabilities
from callGraphUtils import GESCPrinters

root_dir = '/workspaces/solidity'

'''获取完整call图'''
def get_full_call_graph(vulnerabilities_info):
    bug_full_graph = None

    for file_item in vulnerabilities_info:

        # 获取solc版本信息
        version = file_item['version']
        # 设置为当前版本
        command = f"solc-select use {version}"
        subprocess.run(command, shell=True)
        # 获取文件路径
        sol_file_path = file_item['path']
        # 获取文件名称
        sol_file_name = file_item['name']
        # 使用slither解析
        slither = Slither(sol_file_path)

        # 获取文件全部漏洞信息
        list_sol_file_vul_info = get_vulnerabilities(sol_file_name, vulnerabilities_info)

        # 初始化call生成器类
        call_graph_printer = GESCPrinters(slither, None)
        # 生成call图
        all_contracts_call_graph = call_graph_printer.generate_call_graph(sol_file_name, list_sol_file_vul_info)

        # 添加到全局call图
        if bug_full_graph is None:
            bug_full_graph = deepcopy(all_contracts_call_graph)
        elif all_contracts_call_graph is not None:
            bug_full_graph = nx.disjoint_union(bug_full_graph, all_contracts_call_graph)
    
    return bug_full_graph

'''获取单种漏洞图'''
def get_vuln_call_graph(vuln_dataset_dir, isSave=False):
        
    vuln_json_path = os.path.join(vuln_dataset_dir, 'vulnerabilities.json')
    with open(vuln_json_path, 'r') as f:
        vulnerabilities_info = list(json.load(f))
    bug_full_graph = get_full_call_graph(vulnerabilities_info)
    # ##########################################
    from utils import check_null
    if not check_null(bug_full_graph):
        print('有空结点')
    else:
        print('无空结点')
    ##########################################
    if isSave:
        # 保存图
        nx.write_gpickle(bug_full_graph, os.path.join(vuln_dataset_dir, 'cg.gpickle'))


''' 测试单个文件'''
if __name__ == "__main__":

    dataset_root = f'{root_dir}/integrate_dataset'
    parser = argparse.ArgumentParser()
    parser.add_argument('--isSave', action='store_true', help='是否保存生成图')
    # parser.add_argument('--vuln_type', default='other', type=str, help='检测漏洞类型')
    args = parser.parse_args()

    isSave = args.isSave

    # 获取全部漏洞类型
    all_vuln_type = [x for x in os.listdir(dataset_root) if x != 'clean']
    # all_vuln_type = [args.vuln_type]
    # print(all_vuln_type)
    # 对每一种漏洞类型进行处理
    for vuln_type in all_vuln_type:
        vuln_dataset_dir = os.path.join(dataset_root, vuln_type, 'integrate')
        get_vuln_call_graph(vuln_dataset_dir, isSave=isSave)