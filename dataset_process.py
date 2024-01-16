import os
import json
import subprocess

root_path = '/workspaces/solidity'

def mkdir_vuln_dataset(target_dir, vuln_dir):
    for dataset in vuln_dir:
        for vuln in vuln_dir[dataset]:
            dataset_dir = os.path.join(target_dir, vuln, dataset)
            command = f'mkdir -p {dataset_dir}'
            subprocess.run(command, shell=True)

def migrate_vuln_dataset(source_dir, target_dir, vuln_dir, json_dir):
    for dataset in vuln_dir:
        source_json = os.path.join(json_dir, f'{dataset}.json')
        with open(source_json, 'r') as f:
            items = json.load(f)
        for vuln in vuln_dir[dataset]:
            # 数据转移
            source = os.path.join(source_dir, dataset, vuln) 
            target = os.path.join(target_dir, vuln, dataset)
            command = f'cp -r {source}/* {target}'
            subprocess.run(command, shell=True)

            # 清除所有非.sol文件
            for file_name in os.listdir(target):
                if not file_name.endswith('.sol'):
                    file_path = os.path.join(target, file_name)
                    command = f'rm -r {file_path}'
                    subprocess.run(command, shell=True)
            
            # 修改文件名({dataset}_{vuln}_{file_name})
            for file_name in os.listdir(target):
                old_path = os.path.join(target, file_name)
                new_path = os.path.join(target, f'{dataset}_{vuln}_{file_name}')
                os.rename(old_path, new_path)

            # 划分json修改项
            target_json = os.path.join(target_dir, vuln, dataset, f'{vuln}_{dataset}_vulnerabilities.json')
            items_vuln = []
            for item in items:
                if item['path'].split('/')[-2] == vuln:
                    new_name = '_'.join(item['path'].split('/'))
                    item['name'] = new_name
                    item['path'] = f'{target}/{new_name}'
                    items_vuln.append(item)
            # 存入新文件
            with open(target_json, 'w') as f:
                json.dump(items_vuln, f)

def mkdir_clean_dataset(target_dir):
    target = os.path.join(target_dir, 'clean')
    command = f'mkdir {target}'
    subprocess.run(command, shell=True)

def migrate_clean_dataset(source_dir, target_dir, clean_json):
    target_dataset = os.path.join(target_dir, 'clean')

    with open(clean_json, 'r') as f:
        items = json.load(f)
    
    for item in items:
        source = os.path.join(source_dir, item['path'])
        command = f'cp {source} {target_dataset}'
        subprocess.run(command, shell=True)
        new_name = '_'.join(item['path'].split('/'))
        # 修改名称
        old_path = os.path.join(target_dir, item['path'])
        new_path = os.path.join(target_dataset, new_name)
        os.rename(old_path, new_path)
        item['name'] = new_name
        item['path'] = new_path

    target_json = os.path.join(target_dataset, 'clean_vulnerabilities.json')
    
    with open(target_json, 'w') as f:
        json.dump(items, f)

def get_vuln_info(dataset_dir, dataset_all):
    vuln_all = []
    vuln_dir = {}
    for dataset in dataset_all:
        dataset_path = os.path.join(dataset_dir, dataset)
        dataset_vuln = [x for x in os.listdir(dataset_path) if not x.endswith('.json') and  not x[0]=='.']
        vuln_all.extend(dataset_vuln)
        vuln_dir[dataset] = dataset_vuln
    vuln_all = set(vuln_all)

    return vuln_all, vuln_dir

def mkdir_integrate_dataset(target_dir, vuln_all):
    
    for vuln in vuln_all:
        target = os.path.join(target_dir, vuln, 'integrate')
        command = f'mkdir {target}'
        subprocess.run(command, shell=True)

def integrate_dataset(target_dir, vuln_dir):
    
    for dataset in vuln_dir:
        for vuln in vuln_dir[dataset]:
            source = os.path.join(target_dir, vuln, dataset, f'{vuln}_{dataset}_vulnerabilities.json')
            target = os.path.join(target_dir, vuln, 'integrate', f'{vuln}_{dataset}_vulnerabilities.json')
            command = f'cp {source} {target}'
            subprocess.run(command, shell=True)

def integrate_json(target_dir, vuln_all):

    for vuln in vuln_all:
        vuln_dir = os.path.join(target_dir, vuln, 'integrate')
        items = []

        json_path = [os.path.join(vuln_dir, x) for x in os.listdir(vuln_dir) if x.endswith('.json')]
        for path in json_path:
            with open(path, 'r') as f:
                item = json.load(f)
                items.extend(list(item))
        with open(os.path.join(vuln_dir, 'vuln_vulnerabilities.json'), 'w') as f:
            json.dump(items, f)

''' 测试单个文件'''
if __name__ == "__main__":
    
    dataset_dir = '/workspaces/solidity/dataset'
    dataset_all = ['smartbugs', 'solidifi']
    target_dir = '/workspaces/solidity/integrate_dataset'
    json_dir = '/workspaces/solidity/json'
    clean_json = '/workspaces/solidity/json/clean.json'

    vuln_all, vuln_dir = get_vuln_info(dataset_dir, dataset_all) 

    # print(vuln_all, vuln_dir)

    # # 新建文件夹
    # mkdir_vuln_dataset(target_dir, vuln_dir)
    # mkdir_clean_dataset(target_dir)

    # # 数据集转移
    # migrate_vuln_dataset(dataset_dir, target_dir, vuln_dir, json_dir)
    migrate_clean_dataset(dataset_dir, target_dir, clean_json)

    # # 数据集合并
    # mkdir_integrate_dataset(target_dir, vuln_all)
    # integrate_dataset(target_dir, vuln_dir)
    # integrate_json(target_dir, vuln_all)
