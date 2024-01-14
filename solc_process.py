import os
import json
import subprocess

from slither.slither import Slither

root_path = '/workspaces/solidity'

def try_solc(contracts_json, error_json):
    counts = []
    error_all = []
    for json_file in contracts_json:
        count = 0
        with open(json_file, 'r') as f:
            items = json.load(f)
        for item in items:
            sol_file_path = os.path.join(root_path, 'dataset', item['path'])
            # 解析文件solc版本
            version = item['version']
            # 设置当前版本
            command = f"solc-select use {version}"
            subprocess.run(command, shell=True)
            try:
                slither = Slither(sol_file_path)
            except Exception as e:
                print(e)
                count += 1
                error_all.append(sol_file_path)
                continue
        counts.append(str(count) + ':' + str(len(items)) )
    
    with open(error_json, 'w') as f:
            json.dump(error_all, f)
    
    print(counts)

''' 测试单个文件'''
if __name__ == "__main__":
    # 
    contracts_json = ['/workspaces/solidity/json/solidifi.json', '/workspaces/solidity/json/clean.json', '/workspaces/solidity/json/smartbugs.json']
    error_json = '/workspaces/solidity/json/error.json'
    try_solc(contracts_json, error_json)