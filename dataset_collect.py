import os
import json
import random
import numpy as np

def collect_clean_dataset(dataset_dir, vuln_all_dir, ratio):

    source_path = os.path.join(dataset_dir, 'clean', 'clean_vulnerabilities.json')
    with open(source_path, 'r') as f:
        clean_items = list(json.load(f))
    len_clean = len(clean_items)
    for vuln in vuln_all_dir:
        target_path = os.path.join(dataset_dir, vuln, 'integrate', 'clean_vulnerabilities.json')
        vuln_item_num = vuln_all_dir[vuln]
        clean_choose = random.choices(np.arange(len_clean), k=vuln_item_num*ratio)
        clean_choose_items = []
        for index in clean_choose:
            clean_choose_items.append(clean_items[index])
        with open(target_path, 'w') as f:
            json.dump(clean_choose_items, f)

def concat_json(dataset_dir, vuln_all):
    for vuln in vuln_all:
        all_items = []

        vuln_path = os.path.join(dataset_dir, vuln, 'integrate', 'vuln_vulnerabilities.json')
        clean_path = os.path.join(dataset_dir, vuln, 'integrate', 'clean_vulnerabilities.json')
        target_path = os.path.join(dataset_dir, vuln, 'integrate', 'vulnerabilities.json')

        with open(vuln_path, 'r') as f:
            items = json.load(f)
        all_items.extend(items)

        with open(clean_path, 'r') as f:
            items = json.load(f)
        all_items.extend(items)

        with open(target_path,'w') as f:
            json.dump(all_items, f)

def make_annotaiton(dataset_dir, vuln_all):
    for vuln in vuln_all:
        source_json_path = os.path.join(dataset_dir, vuln, 'integrate', 'vulnerabilities.json')
        target_json_path = os.path.join(dataset_dir, vuln, 'integrate', 'graph_label.json')
        
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

    dataset_dir = '/workspaces/solidity/integrate_dataset'
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
    concat_json(dataset_dir, vuln_all)
    # 生成图标签索引文件
    make_annotaiton(dataset_dir, vuln_all)