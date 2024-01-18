# 构建全局异构图
import os
import json
import argparse	
import subprocess
import networkx as nx
from copy import deepcopy
from slither.slither import Slither
from slither.core.cfg.node import NodeType

from vulnInfo import get_vulnerabilities, get_vuln_of_node

root_dir = '/workspaces/solidity'

'''获取一种漏洞的完整图'''
def get_full_cfg_graph(vulnerabilities_info):
    # smart_contracts 该漏洞全部.sol文件的路径
    # data_vulnerabilities 该漏洞全部漏洞信息

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
        # 提取单个文件图
        sol_file_graph = None
        for contract in slither.contracts:
            # 初始化合约图
            contract_graph = nx.MultiDiGraph()
            # 添加状态变量节点
            for state_var in contract.state_variables:
                node_token = '_'.join([str(sol_file_name),
                                       str(contract.name),
                                       str(state_var.full_name)])
                
                node_code_lines = state_var.source_mapping.lines
                node_vuln_info = get_vuln_of_node(node_code_lines, list_sol_file_vul_info)
                contract_graph.add_node(f'{contract.name}_{state_var.full_name}',
                                        node_type='STATEVARIABLE', 
                                        node_expression=state_var.name,
                                        node_token=node_token,
                                        node_code_lines=node_code_lines,
                                        node_vuln_info=node_vuln_info,
                                        function_fullname=None,
                                        contract_name=contract.name, 
                                        source_file=sol_file_name)
               
            for function in contract.functions + contract.modifiers:
                
                if function.contract_declarer.name != contract.name:
                    # 继承的私有函数不属于当前合约
                    if function.visibility == 'private':
                        continue

                # 存储函数状态变量使用情况
                state_var_use = {}
                # 局部变量字典
                local_var_dict = {}
                # 初始化函数图
                func_graph =  nx.MultiDiGraph()
                
                for node in function.nodes:

                    # 存数使用到的局部变量
                    node_local_var = []

                    # 获取节点信息
                    node_type, node_expression, node_token, node_code_lines, node_vuln_info = get_node_info(node, list_sol_file_vul_info)
                    # 添加节点    
                    func_graph.add_node(node.node_id,
                                        node_type=node_type,
                                        node_expression=node_expression,
                                        node_token=node_token,
                                        node_code_lines=node_code_lines,
                                        node_vuln_info=node_vuln_info,
                                        function_fullname=function.full_name,
                                        contract_name=contract.name, 
                                        source_file=sol_file_name)
                     # 判断节点类型
                    if node.type == NodeType.VARIABLE:
                        # 添加局部变量声明
                        local_var_name = node._variable_declaration
                        local_var_dict[local_var_name] = node.node_id
                    else:
                        # 添加局部变量使用
                        if (node.local_variables_read + node.local_variables_written):
                            node_local_var = [x for x in (node.local_variables_read + node.local_variables_written)]

                        # 添加状态变量
                        for state_var in (function.state_variables_read + function.state_variables_written):
                            if state_var not in state_var_use.keys():
                                state_var_use[state_var] = [node.node_id]
                            else:
                                state_var_use[state_var].append(node.node_id)

                    # 添加控制边
                    if node.type in [NodeType.IF, NodeType.IFLOOP]:
                        true_node = node.son_true
                        if true_node:
                            if true_node.node_id not in func_graph.nodes():
                                # 获取节点信息
                                node_type, node_expression, node_token, node_code_lines, node_vuln_info = get_node_info(true_node, list_sol_file_vul_info)
                                # 添加节点    
                                func_graph.add_node(true_node.node_id,
                                                    node_type=node_type,
                                                    node_expression=node_expression,
                                                    node_token=node_token,
                                                    node_code_lines=node_code_lines,
                                                    node_vuln_info=node_vuln_info,
                                                    function_fullname=function.full_name,
                                                    contract_name=contract.name, 
                                                    source_file=sol_file_name)
                            # 添加边
                            func_graph.add_edge(node.node_id, 
                                                true_node.node_id,
                                                edge_type='if_true')

                        false_node = node.son_false
                        if false_node:
                            if false_node.node_id not in func_graph.nodes():
                                # 获取节点信息
                                node_type, node_expression, node_token, node_code_lines, node_vuln_info = get_node_info(false_node, list_sol_file_vul_info)
                                # 添加节点    
                                func_graph.add_node(false_node.node_id,
                                                    node_type=node_type,
                                                    node_expression=node_expression,
                                                    node_token=node_token,
                                                    node_code_lines=node_code_lines,
                                                    node_vuln_info=node_vuln_info,
                                                    function_fullname=function.full_name,
                                                    contract_name=contract.name, 
                                                    source_file=sol_file_name)
                            # 添加边
                            func_graph.add_edge(node.node_id, 
                                                false_node.node_id,
                                                edge_type='if_false')
                    # 添加顺序边
                    else:
                        for son_node in node.sons:
                            if son_node.node_id not in func_graph.nodes():
                                # 获取节点信息
                                node_type, node_expression, node_token, node_code_lines, node_vuln_info = get_node_info(son_node, list_sol_file_vul_info)
                                # 添加节点    
                                func_graph.add_node(son_node.node_id,
                                                    node_type=node_type,
                                                    node_expression=node_expression,
                                                    node_token=node_token,
                                                    node_code_lines=node_code_lines,
                                                    node_vuln_info=node_vuln_info,
                                                    function_fullname=function.full_name,
                                                    contract_name=contract.name, 
                                                    source_file=sol_file_name)
                            # 添加边
                            func_graph.add_edge(node.node_id,
                                                son_node.node_id,
                                                edge_type='next')
                    
                    # 添加函数局部变量数据边
                    for local_var in node_local_var:
                        if local_var in local_var_dict.keys():
                            func_graph.add_edge(local_var_dict[local_var],
                                                node.node_id,
                                                edge_type='use')
                    
                if len(func_graph.nodes) != 0:
                    # 添加函数名称
                    func_graph = nx.relabel_nodes(func_graph,  \
                                lambda x: f'{contract.name}_{function.full_name}_{str(x)}', copy=False)

                # 添加函数节点
                function_node_token = '_'.join([str(sol_file_name),
                                                str(contract.name),
                                                str(function.full_name)])
                
                function_node_code_lines = function.source_mapping.lines
                function_node_vuln_info = get_vuln_of_node(function_node_code_lines, list_sol_file_vul_info)
                func_graph.add_node(f'{contract.name}_{function.full_name}',
                                    node_type='FUNCTION',
                                    node_expression=None,
                                    node_token=function_node_token,
                                    node_code_lines=function_node_code_lines,
                                    node_vuln_info=function_node_vuln_info,
                                    function_fullname=function.full_name,
                                    contract_name=contract.name, 
                                    source_file=sol_file_name)
                # 添加函数边
                if f'{contract.name}_{function.full_name}_0' in func_graph.nodes():
                    func_graph.add_edge(f'{contract.name}_{function.full_name}', f'{contract.name}_{function.full_name}_0', edge_type='next')
                
                # 合并图
                contract_graph = nx.compose(contract_graph, func_graph)
               
                # 添加全局数据边
                for state_var in state_var_use:
                    if (state_var in contract.state_variables) or state_var.visibility != 'private':
                        for node_id in state_var_use[state_var]:
                            contract_graph.add_edge(f'{contract.name}_{state_var.full_name}',
                                                    f'{contract.name}_{function.full_name}_{str(node_id)}',
                                                    edge_type='use')
                    else:
                        for node_id in state_var_use[state_var]:
                            contract_graph.add_edge(f'{function.contract_declarer.name}_{state_var.full_name}',
                                                    f'{contract.name}_{function.full_name}_{str(node_id)}',
                                                    edge_type='use')

                ##########################################
                # from utils import check_null
                # if not check_null(contract_graph):
                #     print(contract.name)
                #     print('有空结点')
                ##########################################

            # if contract_graph is None:
            #     continue
            if sol_file_graph is None:
                sol_file_graph = deepcopy(contract_graph)
            elif sol_file_graph is not None:
                sol_file_graph = nx.compose(sol_file_graph, contract_graph)
                # sol_file_graph = nx.disjoint_union(sol_file_graph, contract_graph)
            
        ##########################################
        # from utils import check_null
        # if not check_null(sol_file_graph):
        #     print('有空结点')
        # else:
        #     print('无空节点')
        ##########################################

        if bug_full_graph is None:
            bug_full_graph = deepcopy(sol_file_graph)
        elif bug_full_graph is not None:
            bug_full_graph = nx.disjoint_union(bug_full_graph, sol_file_graph)
    
    return bug_full_graph
                    

'''获取单种漏洞图'''
def get_vuln_cfg_graph(vuln_dataset_dir, isSave=False):

    vuln_json_path = os.path.join(vuln_dataset_dir, 'vulnerabilities.json')
    with open(vuln_json_path, 'r') as f:
        vulnerabilities_info = list(json.load(f))
    bug_full_graph = get_full_cfg_graph(vulnerabilities_info)
    # ##########################################
    from utils import check_null
    if not check_null(bug_full_graph):
        print('有空结点')
    else:
        print('无空结点')
    ##########################################
    if isSave:
        # 保存图
        nx.write_gpickle(bug_full_graph, os.path.join(vuln_dataset_dir, 'cfg.gpickle'))

'''获取节点全部信息'''
def get_node_info(node, list_sol_file_vul_info):

    node_type = str(node.type)

    if node.expression:
        node_expression = str(node.expression)
    else:
        if node.variable_declaration:
            node_expression = str(node.variable_declaration)
        else:
            node_expression = None

    node_token = "_".join([node_type, str(node_expression)])

    node_code_lines = node.source_mapping.lines

    node_vuln_info = get_vuln_of_node(node_code_lines, list_sol_file_vul_info)
    
    return node_type, node_expression, node_token, node_code_lines, node_vuln_info

def check_inheriate(function, contract_name):
    # 继承的私有函数不属于当前合约
    if function.visibility == 'private':
        return False
    # 继承的函数中访问private变量
    state_varibales = function.state_variables_read + function.state_variables_written
    for var in state_varibales:
        if var.visibility == 'private' and contract_name != var.contract:
            return False
    return True

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
    # all_vuln_type = ['unchecked_low_level_calls']
    # 对每一种漏洞类型进行处理
    for vuln_type in all_vuln_type:
        vuln_dataset_dir = os.path.join(dataset_root, vuln_type, 'integrate')
        get_vuln_cfg_graph(vuln_dataset_dir, isSave=isSave)