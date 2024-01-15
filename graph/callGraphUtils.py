import os
import json
import subprocess
import networkx as nx
from collections import defaultdict

from slither.slither import Slither
from slither.printers.abstract_printer import AbstractPrinter
from slither.core.declarations.function import Function
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.variables.variable import Variable

from vulnInfo import get_vulnerabilities, get_vuln_of_node

root_dir = '/workspaces/solidity' 
'''返回边元组'''
def _edge(from_node, to_node, edge_type):
    return (from_node, to_node, edge_type)

class GESCPrinters(AbstractPrinter):

    ARGUMENT = "call-graph"
    HELP = "Export the call-graph of the contracts to a dot file"
    WIKI = "https://github.com/trailofbits/slither/wiki/Printer-documentation#call-graph"

    def __init__(self, slither, logger=None):
        super().__init__(slither, logger)

    def generate_call_graph(self, filename, vulnerabilities_in_sc):

        all_functionss = [
            compilation_unit.functions for compilation_unit in self.slither.compilation_units
        ]
        all_functions = [item for sublist in all_functionss for item in sublist]
        all_functions_as_dict = {
            function.canonical_name: function for function in all_functions
        }

        call_graph = _process_functions(all_functions_as_dict.values(), 
                                        filename, 
                                        vulnerabilities_in_sc)

        return call_graph

    def output(self, filename):
        """
        Output the graph in filename
        """
        pass

def _process_functions(functions, filename_input, vulnerabilities_in_sc=None):

    contract_functions = defaultdict(set)  # contract -> contract functions nodes
    contract_calls = defaultdict(set)  # contract -> contract calls edges

    solidity_functions = set()  # solidity function nodes
    solidity_calls = set()  # solidity calls edges
    external_calls = set()  # external calls edges

    all_contracts = set()

    for function in functions:
        all_contracts.add(function.contract_declarer)

    for function in functions:
        _process_function(
            function.contract_declarer,
            function,
            contract_functions,
            contract_calls,
            solidity_functions,
            solidity_calls,
            external_calls,
            all_contracts,
            filename_input,
            vulnerabilities_in_sc
        )

    # print('contract_functions:', contract_functions)
    # print('solidity_functions:', solidity_functions)
    # print('contract_calls:', contract_calls)
    # print('solidity_calls:', solidity_calls)
    # print('external_calls:', external_calls)
    # print('all_contracts:', all_contracts)

    all_contracts_graph = nx.MultiDiGraph()
    for contract in all_contracts:
        _render_internal_calls(all_contracts_graph, contract,
                               contract_functions, contract_calls)
    
    # _render_solidity_calls(all_contracts_graph, solidity_functions, solidity_calls)
    _render_external_calls(all_contracts_graph, external_calls)

    return all_contracts_graph

def _process_function(
    contract,
    function,
    contract_functions,
    contract_calls,
    solidity_functions,
    solidity_calls,
    external_calls,
    all_contracts,
    filename_input,
    vulnerabilities_in_sc=[]
):  
    tuple_vulnerabilities_in_sc = parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc)
    contract_functions[contract].add(tuple(
        _function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items())
    )

    for internal_call in function.internal_calls:
        _process_internal_call(
            contract,
            function,
            internal_call,
            contract_calls,
            solidity_functions,
            solidity_calls,
            filename_input,
            vulnerabilities_in_sc
        )

    for external_call in function.high_level_calls:
        _process_external_call(
            contract,
            function,
            external_call,
            contract_functions,
            external_calls,
            all_contracts,
            filename_input,
            vulnerabilities_in_sc
        )

''''将文件的漏洞信息list和字典转化为tuple'''
def parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc):
    vul_info = list()

    if vulnerabilities_in_sc is not None:
        for vul in vulnerabilities_in_sc:
            for key, value in vul.items():
                if key == 'lines':
                    # 将lines类型由list转化为tuple
                    vul[key] = tuple(value)
        
        for vul in vulnerabilities_in_sc:
            vul_info.append(tuple(vul.items()))

    vul_info = tuple(vul_info)

    return vul_info

'''获取函数节点的唯一表示'''
def _function_node(contract, function, filename, tuple_vulnerabilities_in_sc):
    
    # 函数在文件的位置
    node_function_source_code_lines = function.source_mapping.lines
    # 获取文件漏洞信息
    vulnerabilities_in_sc = revert_vulnerabilities_in_sc_from_tuple(tuple_vulnerabilities_in_sc)
    # 获取节点漏洞信息
    node_function_info_vulnerabilities = get_vuln_of_node(node_function_source_code_lines, vulnerabilities_in_sc)

    node_info = {'node_id':f"{filename}_{contract.id}_{contract.name}_{function.full_name}",
                 'node_token':f'{filename}_{contract.name}_{function.full_name}',
                 'node_code_lines':tuple(node_function_source_code_lines),
                 'node_vuln_info':parse_vulnerabilities_in_sc_to_tuple(node_function_info_vulnerabilities),
                 'function_fullname':function.full_name,
                 'contract_name':contract.name, 
                 'source_file':filename,
                 }

    return node_info

'''获取solidity函数节点的节点表示'''
def _solidity_function_node(solidity_function):
    node_info = {
        'node_id': f"[Solidity]_{solidity_function.full_name}",
        'node_token':f"[Solidity]_{solidity_function.full_name}",
        'node_code_lines':None,
        'node_vuln_info':None,
        'function_fullname': solidity_function.full_name,
        'contract_name': None,
        'source_file': None,
    }
    return node_info

'''输出[{},{}]形式的文件漏洞信息'''
def revert_vulnerabilities_in_sc_from_tuple(tuple_vulnerabilities_in_sc):
    # tuple_vulnerabilities_in_sc 为tuple形式的文件漏洞信息

    # 转化为list
    vulnerabilities_in_sc = list(tuple_vulnerabilities_in_sc)
    vul_info = []
    if len(vulnerabilities_in_sc) > 0:
        for vul in vulnerabilities_in_sc:
            dct = dict((x, y) for x, y in vul)
            for key, val in dct.items():
                if key == 'lines':
                    dct[key] = list(val)

            vul_info.append(dct)
    
    return vul_info

'''处理内部调用'''
def _process_internal_call(
    contract,
    function,
    internal_call,
    contract_calls,
    solidity_functions,
    solidity_calls,
    filename_input,
    vulnerabilities_in_sc=[]
):
    tuple_vulnerabilities_in_sc = parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc)
    if isinstance(internal_call, (Function)):
        # print('tuple:', tuple(_function_node(contract, function, filename_input).items()))
        contract_calls[contract].add(
            _edge(
                tuple(_function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items()),
                tuple(_function_node(contract, internal_call, filename_input, tuple_vulnerabilities_in_sc).items()),
                edge_type='internal_call',
            )
        )

    elif isinstance(internal_call, (SolidityFunction)):
        solidity_functions.add(tuple(_solidity_function_node(internal_call).items()))
        solidity_calls.add(
            _edge(
                tuple(_function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items()),
                tuple(_solidity_function_node(internal_call).items()),
                edge_type='solidity_call',
            )
        )

'''处理外部调用'''
def _process_external_call(
    contract,
    function,
    external_call,
    contract_functions,
    external_calls,
    all_contracts,
    filename_input,
    vulnerabilities_in_sc=[]
):
    tuple_vulnerabilities_in_sc = parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc)
    external_contract, external_function = external_call
    
    if not external_contract in all_contracts:
        return

    # add variable as node to respective contract
    if isinstance(external_function, (Variable)):
        contract_functions[external_contract].add(tuple(
                _function_node(external_contract, external_function, filename_input, tuple_vulnerabilities_in_sc).items()))

    external_calls.add(
        _edge(
            tuple(_function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items()),
            tuple(_function_node(external_contract, external_function, filename_input, tuple_vulnerabilities_in_sc).items()),
            edge_type='external_call'
        )
    )

'''添加函数节点和内部调用边'''
def _render_internal_calls(nx_graph, contract, contract_functions, contract_calls):
    
    if len(contract_functions[contract]) > 0:
        for contract_function in contract_functions[contract]: 

            # 获取节点信息
            node_id, node_type, node_token, node_code_lines, node_vuln_info, function_fullname, contract_name, source_file= get_node_info(contract_function)
            # 添加节点
            nx_graph.add_node(node_id,
                              node_type=node_type,
                              node_token=node_token,
                              node_code_lines=node_code_lines,
                              node_vuln_info=node_vuln_info,
                              function_fullname=function_fullname,
                              contract_name=contract_name, 
                              source_file=source_file,
                            )
                            
    # 添加内部调用边
    if len(contract_calls[contract]) > 0:
        for contract_call in contract_calls[contract]:
            add_edge_info(contract_call, nx_graph)

'''添加外部调用边'''
def _render_external_calls(nx_graph, external_calls):
    if len(external_calls) > 0:
        for external_call in external_calls:
            add_edge_info(external_call, nx_graph)

'''获取节点信息'''
def get_node_info(tuple_node):

    # 'node_id': f"[Solidity]_{solidity_function.full_name}",
    # 'node_token':f"[Solidity]_{solidity_function.full_name}",
    # 'node_code_lines':None,
    # 'node_vuln_info':None,
    # 'function_fullname': solidity_function.full_name,
    # 'contract_name': None,
    # 'source_file': None,

    if tuple_node[0][0] == 'node_id':
        node_id = tuple_node[0][1]
    if tuple_node[1][0] == 'node_token':
        node_token = tuple_node[1][1]
    if tuple_node[2][0] == 'node_code_lines':
        node_code_lines = list(tuple_node[2][1])
    if tuple_node[3][0] == 'node_vuln_info':
        node_vuln_info = revert_vulnerabilities_in_sc_from_tuple(tuple_node[3][1])
    if tuple_node[4][0] == 'function_fullname':
        function_fullname = tuple_node[4][1]
    if tuple_node[5][0] == 'contract_name':
        contract_name = tuple_node[5][1]
    if tuple_node[6][0] == 'source_file':
        source_file = tuple_node[6][1]
    
    if len(node_vuln_info) == 0:
        node_vuln_info = None

    if 'fallback' in node_id:
        node_type = 'fallback_function'
    elif '[Solidity]' in node_id:
        node_type = 'fallback_function'
    else:
        node_type = 'contract_function'

    return node_id, node_type, node_token, node_code_lines, node_vuln_info, function_fullname, contract_name, source_file


'''添加边信息'''
def add_edge_info(contract_call, nx_graph):
    # 获取起始节点信息
    source = contract_call[0]
    source_node_id, node_type, node_token, node_code_lines, node_vuln_info, function_fullname, contract_name, source_file = get_node_info(source)

    if source_node_id not in nx_graph.nodes():
        nx_graph.add_node(source_node_id,
                          node_type=node_type,
                          node_token=node_token,
                          node_code_lines=node_code_lines,
                          node_vuln_info=node_vuln_info,
                          function_fullname=function_fullname,
                          contract_name=contract_name, 
                          source_file=source_file,
                        )

    # 获取目标节点信息
    target = contract_call[1]
    target_node_id, node_type, node_token, node_code_lines, node_vuln_info, function_fullname, contract_name, source_file = get_node_info(target)

    if target_node_id not in nx_graph.nodes():
        nx_graph.add_node(target_node_id,
                          node_type=node_type,
                          node_token=node_token,
                          node_code_lines=node_code_lines,
                          node_vuln_info=node_vuln_info,
                          function_fullname=function_fullname,
                          contract_name=contract_name, 
                          source_file=source_file,
                        )

    edge_type = contract_call[2]

    nx_graph.add_edge(source_node_id, target_node_id, edge_type=edge_type)

''' 测试单个文件'''
if __name__ == "__main__":

    json_path = '/workspaces/solidity/integrate_dataset/other/integrate/vulnerabilities.json'
    with open(json_path, 'r') as f:
        items = json.load(f)
    file_item = items[0]

    # 文件路径
    sol_file_path = file_item['path']
    # 漏洞信息
    vulnerabilities_info = items


    # solc文件名称
    sol_file_name = file_item['name']
    # 文件solc版本
    version = file_item['version']
    # 设置当前版本
    command = f"solc-select use {version}"
    subprocess.run(command, shell=True)

    # 使用对应版本的slither解析
    slither = Slither(sol_file_path)
    # 获取文件漏洞信息
    list_sol_file_vul_info = get_vulnerabilities(sol_file_name, vulnerabilities_info)
    
    # 初始化call生成器类
    call_graph_printer = GESCPrinters(slither, None)
    # 生成call图
    all_contracts_call_graph = call_graph_printer.generate_call_graph(sol_file_name, list_sol_file_vul_info)
    print(all_contracts_call_graph.nodes())
    print(all_contracts_call_graph.edges())
