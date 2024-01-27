import os
import json
import random
import subprocess
import numpy as np

def collect_clean_dataset(dataset_dir, vuln_all_dir, ratio):

    source_path = os.path.join(dataset_dir, 'clean', 'clean_vulnerabilities.json')
    with open(source_path, 'r') as f:
        clean_items = list(json.load(f))
    len_clean = len(clean_items)

    for vuln in vuln_all_dir:
        command = f"mkdir -p {os.path.join(dataset_dir, vuln, 'integrate', str(ratio))}"
        subprocess.run(command, shell=True)
        target_path = os.path.join(dataset_dir, vuln, 'integrate', str(ratio), 'clean_vulnerabilities.json')
        # target_path = os.path.join(dataset_dir, vuln, 'clean_vulnerabilities.json')
        vuln_item_num = vuln_all_dir[vuln]
        # np.random.choice(a=np.arange(5), size=5, replace=False, p=None)
        # clean_choose = random.choices(np.arange(len_clean), k=vuln_item_num*ratio)
        clean_choose = np.random.choice(a=np.arange(len_clean), size=vuln_item_num*ratio, replace=False, p=None)
        # print(clean_choose)
        clean_choose_items = []
        for index in clean_choose:
            clean_choose_items.append(clean_items[index])
        with open(target_path, 'w') as f:
            json.dump(clean_choose_items, f)

def concat_json(dataset_dir, vuln_all, ratio):
    for vuln in vuln_all:
        all_items = []
        command = f"mkdir -p {os.path.join(dataset_dir, vuln, 'integrate', ratio)}"
        subprocess.run(command, shell=True)
        vuln_path = os.path.join(dataset_dir, vuln, 'integrate', 'vuln_vulnerabilities.json')
        clean_path = os.path.join(dataset_dir, vuln, 'integrate', ratio, 'clean_vulnerabilities.json')
        target_path = os.path.join(dataset_dir, vuln, 'integrate', ratio, 'vulnerabilities.json')

        with open(vuln_path, 'r') as f:
            items = json.load(f)
        all_items.extend(items)

        with open(clean_path, 'r') as f:
            items = json.load(f)
        all_items.extend(items)

        with open(target_path,'w') as f:
            json.dump(all_items, f)

def make_annotaiton(dataset_dir, vuln_all, ratio):
    for vuln in vuln_all:
        command = f"mkdir -p {os.path.join(dataset_dir, vuln, 'integrate', ratio)}"
        subprocess.run(command, shell=True)

        source_json_path = os.path.join(dataset_dir, vuln, 'integrate', ratio, 'vulnerabilities.json')
        target_json_path = os.path.join(dataset_dir, vuln, 'integrate', ratio, 'graph_label.json')
        
        target = []
        with open(source_json_path, 'r') as f:
            source = json.load(f)
        for item in source:
            item_dict = {}
            item_dict['contract_name'] = item['name']
            if 'vulnerabilities' not in item.keys():
                item_dict['targets'] = 0
            else:
                item_dict['targets'] = 1
            target.append(item_dict)

        with open(target_json_path, 'w') as f:
            json.dump(target, f)


if __name__ == "__main__":

    root_path = '/workspaces/solidity'
    dataset_dir = os.path.join(root_path, 'integrate_dataset')
    # dataset_dir = '/workspaces/solidity/integrate_dataset'
    vuln_all = [x for x in os.listdir(dataset_dir) if x != 'clean']
    vuln_all_dir = {}
    for vuln in vuln_all:
        path = os.path.join(dataset_dir, vuln, 'integrate', 'vuln_vulnerabilities.json') 
        with open(path, 'r') as f:
            len_item = len(json.load(f))
        vuln_all_dir[vuln] = len_item
    
    ratio = 1

    # 为每种漏洞选择正样本1:1
    collect_clean_dataset(dataset_dir, vuln_all_dir, ratio)
    # 合并json文件
    concat_json(dataset_dir, vuln_all, str(ratio))
    # 生成图标签索引文件
    make_annotaiton(dataset_dir, vuln_all, str(ratio))