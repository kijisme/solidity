'''获取文件全部漏洞信息'''
def get_vulnerabilities(file_name, vulnerabilities):
    # 如果为clean 返回None（不存在vulnerabilities键）
    list_vulnerability_in_sc = None
    if vulnerabilities is not None:
        for vul_item in vulnerabilities:
            if file_name == vul_item['name'] and 'vulnerabilities' in vul_item.keys():
                list_vulnerability_in_sc = vul_item['vulnerabilities']
            
    return list_vulnerability_in_sc

'''获取节点对应的漏洞列表'''
def get_vuln_of_node(node_code_lines, list_sol_file_vul_info):
    if list_sol_file_vul_info is not None:
        list_vulnerability = []
        for vul_info_sc in list_sol_file_vul_info:
            vulnerabilities_lines = vul_info_sc['lines']
            interset_lines = set(vulnerabilities_lines).intersection(set(node_code_lines))
            if len(interset_lines) > 0:
                list_vulnerability.append(vul_info_sc)
    else:
        list_vulnerability = None
    
    if list_vulnerability is None or len(list_vulnerability) == 0:
        node_info_vulnerabilities = None
    else:
        node_info_vulnerabilities = list_vulnerability

    return node_info_vulnerabilities
