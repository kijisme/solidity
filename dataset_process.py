import os
import json
import subprocess

root_path = '/workspaces/solidity'

def mkdir_vuln_dataset(target_dir, vuln_all, dataset_all):
    for vuln in vuln_all:
        vuln_dir = os.path.join(target_dir, vuln)
        command = f'mkdir {vuln_dir}'
        subprocess.run(command, shell=True)
        for dataset in dataset_all:
            dataset_dir = os.path.join(vuln_dir, dataset)
            command = f'mkdir {dataset_dir}'
            subprocess.run(command, shell=True)

def migrate_vuln_dataset(dataset_dir, target_dir, vuln_all, vuln_dir):
    # 复制全部文件
    for dataset in os.listdir(dataset_dir):
        source_json = os.path.join(dataset_dir, dataset, 'vulnerabilities.json')
        with open(source_json, 'r') as f:
            items = json.load(f)
        for vuln in vuln_all:    
            if vuln in vuln_dir[dataset]:
                items_vuln = []
                source = os.path.join(dataset_dir, dataset, vuln) 
                target = os.path.join(target_dir, vuln, dataset)
                command = f'cp -r {source} {target}'
                subprocess.run(command, shell=True)

                target_json = os.path.join(target_dir, vuln, dataset, 'vulnerabilities.json')
                for item in items:
                    if item['path'].split('/')[-2] == vuln:
                        items_vuln.append(item)
                with open(target_json, 'w') as f:
                    json.dump(items_vuln, f)

def mkdir_clean_dataset(target_dir):
    target = os.path.join(target_dir, 'clean')
    command = f'mkdir {target}'
    subprocess.run(command, shell=True)

def migrate_clean_dataset(dataset_dir, target_dir, clean_json):
    target_dir = os.path.join(target_dir, 'clean')

    with open(clean_json, 'r') as f:
        items = json.load(f)
    
    for item in items:
        source = os.path.join(dataset_dir, item['path'])
        command = f'cp {source} {target_dir}'
        subprocess.run(command, shell=True)

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


def mkdir_integrate_dataset(target_dir):
    target = os.path.join(target_dir, 'integrate')
    command = f'mkdir {target}'
    subprocess.run(command, shell=True)

def integrate_dataset(target_dir, vuln_all, vuln_dir):
    target = os.path.join(target_dir, 'integrate')
    for vuln_dir


# def integrate_json(target_dir):
#     target = 

''' 测试单个文件'''
if __name__ == "__main__":
    
    dataset_dir = '/workspaces/solidity/dataset'
    dataset_all = ['smartbugs', 'solidifi']
    target_dir = '/workspaces/solidity/integrate_dataset'

    vuln_all, vuln_dir = get_vuln_info(dataset_dir, dataset_all) 

    clean_json = '/workspaces/solidity/json/clean.json'

    print(vuln_all)

    # mkdir_vuln_dataset(target_dir, vuln_all, dataset_all)
    # mkdir_integrate_dataset(target_dir)
    # migrate_clean_dataset(dataset_dir, target_dir, clean_json)

    # migrate_vuln_dataset(dataset_dir, target_dir, vuln_all, vuln_dir)
    # migrate_clean_dataset(dataset_dir, target_dir, clean_json)