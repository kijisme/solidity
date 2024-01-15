import re
import os
import json
import tqdm
import subprocess
import pandas as pd

root_path = '/workspaces/solidity'

''' 获得当前文件的solc版本'''
def get_solc_version(sol_file_path):

    pattern =  re.compile(r'\d.\d.\d+')

    with open(sol_file_path, 'r') as f:
        line = f.readline()
        while line:
            if 'pragma solidity' in line:
                if len(pattern.findall(line)) > 0:
                    return pattern.findall(line)[0]
                else:
                    return 'None'
            line = f.readline()

    return 'None'

'''处理clean'''
def process_clean_source(clean_dir):
    all_clean = []
    for contract_name in os.listdir(clean_dir):
        if contract_name.endswith('.sol'):
            all_clean.append({  'name':contract_name,
                                'path':os.path.join('clean', contract_name)})
    with open(os.path.join(clean_dir, 'vulnerabilities.json'), 'w') as outfile:
        json.dump(all_clean, outfile)

'''处理solidifi'''
def process_solidifi_source(solidifi_dir):

    vuln_type_list = [f for f in os.listdir(solidifi_dir) if not f.endswith('.json')]
    # 存储漏洞信息
    vulnerabilities = []
    
    for vuln_type in vuln_type_list:
        # 当前漏洞根目录
        vuln_dir = os.path.join(solidifi_dir, vuln_type)
        # 获取全部csv文件
        csv_smart_contracts = [os.path.join(vuln_dir, f) for f in os.listdir(vuln_dir) if f.endswith('.csv')]
        for csv_sc in csv_smart_contracts:
            
            # 完整路径
            sol_path = os.path.splitext(csv_sc)[0].replace('BugLog', 'buggy') + '.sol'
            sol_path = '/'.join(sol_path.split('/')[-3:])
            # 名称
            sol_name = os.path.basename(sol_path)

            # 获取漏洞信息
            bug_lines = []

            df_csv_sc = pd.read_csv(csv_sc)
            for index, row in df_csv_sc.iterrows():
                for i in range(row['length']):
                    bug_lines.append(row['loc'] + i)

            dict_sc_info = {
                'name': sol_name,
                'path': sol_path,
                'source': '',
                'vulnerabilities': [
                    {
                        'lines': bug_lines,
                        'category': vuln_type
                    }
                ]
            }
            vulnerabilities.append(dict_sc_info)

        # 保存全部漏洞信息
        with open(os.path.join(solidifi_dir, 'vulnerabilities.json'), 'w') as outfile:
            json.dump(vulnerabilities, outfile, indent=4)

def process_smartbugs_source(smartbugs_dir):
    # json文件路径
    smartbugs_json = os.path.join(smartbugs_dir, 'vulnerabilities.json')

    # 获取文件
    with open(smartbugs_json, 'r') as f:
        items = json.load(f)

    # 遍历所有条目
    for item in items:
        item['path'] = os.path.join('smartbugs','/'.join(item['path'].split('/')[-2:]))

    # 保存全部信息
    with open(os.path.join(smartbugs_json), 'w') as outfile:
        json.dump(items, outfile)

def process_source(clean_dir, solidifi_dir, smartbugs_dir):

    # 处理clean
    process_clean_source(clean_dir)
    # 合并solidifi
    process_solidifi_source(solidifi_dir)
    # 处理smartbugs
    process_smartbugs_source(smartbugs_dir)

def install_solc(sol_version_list):
    for version in sol_version_list:
        command = f"solc-select install {version}"
        subprocess.run(command, shell=True)

''' 测试单个文件'''
if __name__ == "__main__":

    pattern = r'^\d+\.\d+\.\d+$'

    clean_dir = '/workspaces/solidity/dataset/clean' # [156:2742] [480:2742]  2612
    solidifi_dir = '/workspaces/solidity/dataset/solidifi' # [0:350] [48:350] 
    smartbugs_dir = '/workspaces/solidity/dataset/smartbugs' # [0:143] [13:143]

    all_dir = [clean_dir, solidifi_dir, smartbugs_dir]
    #  clean_dir,, solidifi_dirsmartbugs_dir, solidifi_dir
    output_dir = '/workspaces/solidity/json'

    error_file = {}
    all_versions = set()
    # 加载内部文件    
    for dir in all_dir:
        dataset_name = dir.split('/')[-1]
        count = 0
        error_file[dir.split('/')[-1]] = []
        sol_json_path = os.path.join(dir, 'vulnerabilities.json')
        output_json_path = os.path.join(output_dir, dir.split('/')[-1] + '.json')
        # 读取文件
        with open(sol_json_path, 'r') as f:
            json_file = json.load(f)
        
        # 添加信息
        for item in json_file:
            sol_file_path = os.path.join(root_path, 'dataset', item['path'])
            item['version'] = get_solc_version(sol_file_path)
            if item['version'] == 'None':
                count = count + 1
                if dataset_name == 'smartbugs':
                    item['version'] = '0.4.25'
                elif dataset_name == 'solidifi':
                    item['version'] = '0.5.11'
                elif dataset_name == 'clean':
                    item['version'] = '0.4.25'
                error_file[dir.split('/')[-1]].append(item)
            else:
                if not re.match(pattern, item['version']):
                    item['version'] = '0.4.25'
                elif item['version'].split('.')[0] != '0' or item['version'].split('.')[1] > '8':
                    item['version'] = '0.4.25'
                elif item['version'].split('.')[-1] == '00':
                    item['version'] =  '.'.join(item['version'].split('.')[:-1] + ['0'])
                all_versions.add(item['version'])
        print(count,':',len(json_file))

        # 存储文件
        with open(output_json_path, 'w') as f:
            json.dump(json_file, f)

    install_solc(all_versions)
    print(all_versions)