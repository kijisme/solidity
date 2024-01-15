import os
import json
import subprocess

from slither.slither import Slither

root_path = '/workspaces/solidity'

def try_solc(json_file, error_json):

    count = 0
    dataset = json_file.split('/')[-1].split('.')[0]
    if dataset == 'clean':
        error_all = []

    # 打开对应json文件
    with open(json_file, 'r') as f:
        items = json.load(f)

    # 遍历获取文件或者修改文件
    for idx, item in enumerate(items):
        sol_file_path = os.path.join(root_path, 'dataset', item['path'])
        # 解析文件solc版本
        version = item['version']
        # 设置当前版本
        command = f"solc-select use {version}"
        subprocess.run(command, shell=True)
        try:
            slither = Slither(sol_file_path)
        except Exception as e_1:
            if dataset == 'clean':
                # 添加错误条目
                error_all.append(item['path'])
                # 删除
                del items[idx]
                count += 1
                continue
            else:
                # 设置当前版本
                if dataset == 'solidifi':
                    version_new = '0.5.11'
                elif dataset == 'smartbugs':
                    version_new = '0.4.25'
                command = f"solc-select use {version_new}"
                subprocess.run(command, shell=True)
                try:
                    slither = Slither(sol_file_path)
                    item['version'] = version_new
                except Exception as e:
                    print(e)
                    count += 1
                    continue

    if dataset == 'clean':
        # 保存错误文件
        with open(error_json, 'w') as f:
            json.dump(error_all, f)
    # 存储文件
    with open(json_file, 'w') as f:
        json.dump(items, f)

    return str(count) + ':' + str(len(items))

''' 测试单个文件'''
if __name__ == "__main__":
    # ,'/workspaces/solidity/json/clean.json','/workspaces/solidity/json/solidifi.json', '/workspaces/solidity/json/solidifi.json'
    contracts_jsons = ['/workspaces/solidity/json/clean.json', '/workspaces/solidity/json/solidifi.json', '/workspaces/solidity/json/smartbugs.json']
    error_json = '/workspaces/solidity/json/error.json'
    result = []
    for contracts_json in contracts_jsons:
        result.append(try_solc(contracts_json, error_json))
    print(result)