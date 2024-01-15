import os
import json
import subprocess
from slither.slither import Slither

from solc_install import get_solc_version

root_path = '/workspaces/solidity'

def get_all_error(error_json, target_dir):
    with open(error_json, 'r') as f:
        items = json.load(f)
    for item in items:
        isClean = item.split('/')[-2] == 'clean'
        isSolidifi = item.split('/')[-3] == 'solidifi'

        target = os.path.join(target_dir, ('clean' if isClean else 'solidifi' if isSolidifi else 'smartbugs') +'/' )
        command = f"cp {item} {target}"
        subprocess.run(command, shell=True)

def process_file(file_path, version):
    # 设置当前版本
    command = f"solc-select use {version}"
    subprocess.run(command, shell=True)
    # 解析
    try:
        slither = Slither(file_path)
        return True
    except Exception as e_1:
        return False


''' 测试单个文件'''
if __name__ == "__main__":

    # error_json = '/workspaces/solidity/json/error.json'
    # target_dir = '/workspaces/solidity/dataset/error'
    # get_all_error(error_json, target_dir)

    # dir = '/workspaces/solidity/dataset/error/clean'
    # for file_name in os.listdir(dir):
    #     file_path = os.path.join(dir, file_name)
    #     version = get_solc_version(file_path)
    #     print(f'####{file_name}####')
    #     if version == 'None':
    #         print('version_problem')
    #         version = '0.5.11'
    #     process_file(file_path, version)

    json_dir = '/workspaces/solidity/json'
    result = []
    # 根据json中的version进行编译测试
    for dataset_name in os.listdir(json_dir):
        if dataset_name == 'clean.json' or dataset_name == 'error.json':
            continue
        
        # 打开文件
        count = 0
        with open(os.path.join(json_dir, dataset_name), 'r') as f:
            items = json.load(f)
        for item in items:
            file_path = os.path.join(root_path, 'dataset', item['path'])
            version = item['version']
            flag = process_file(file_path, version)
            if not flag:
                count = count + 1
        result.append(dataset_name + '--' + str(count) + ':' + str(len(items)))

    print(result)

    