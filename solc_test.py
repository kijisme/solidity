import os
import json
import subprocess
from slither.slither import Slither

root_path = '/workspaces/solidity'

def test_contract(file_item):
    version = file_item['version']
    path = file_item['path']
    # 设置当前版本
    # version = '0.5.11'
    command = f"solc-select use {version}"
    subprocess.run(command, shell=True)
    
    try:
        slither = Slither(path)
        return False
    except Exception as e:
        # print(e)
        # print(f'【error】{path}，{version}')
        return True
    
def test_all_contract(json_path):
    
    all_error = []

    with open(json_path, 'r') as f:
        items = json.load(f)
    for file_item in items:
        isError = test_contract(file_item)
        if isError:
            all_error.append(file_item)
    return all_error

''' 测试单个文件'''
if __name__ == "__main__":
    dataset_dir = '/workspaces/solidity/integrate_dataset'
    error_json = '/workspaces/solidity/error.json'
    # vuln_all = [x for x in os.listdir(dataset_dir)]

    vuln_all = ['clean']

    error_all = []
    for vuln in vuln_all:
        if vuln != 'clean':
            json_path = os.path.join(dataset_dir, vuln, 'integrate/vuln_vulnerabilities.json')
            all_error = test_all_contract(json_path)
            error_all.append(all_error)
        else:
            json_path = os.path.join(dataset_dir, vuln, 'clean_vulnerabilities.json')
            all_error = test_all_contract(json_path)
            error_all = all_error
    
    for item in error_all:
        print(f'{len(item)}')

    with open(error_json, 'w') as f:
        json.dump(error_all, f)

    # json_path = '/workspaces/solidity/error.json'
    
    # with open(json_path, 'r') as f:
    #     print(len(json.load(f)))

    # all_error = test_all_contract(json_path)
    # print(len(all_error))
    