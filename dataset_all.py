# 全部漏洞的合并
import os
import json

def integrate_all_vuln(dataset_dir, target, ratio):
    
    all_vuln_items = []
    all_clean_items = []
    all_items = []
    all_label_items = []
    all_vuln = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if x != 'clean']
    for vuln in all_vuln:
        vuln_json = os.path.join(vuln, 'integrate', 'vuln_vulnerabilities.json')
        clean_json = os.path.join(vuln, 'integrate', ratio, 'clean_vulnerabilities.json')
        all_json = os.path.join(vuln, 'integrate', ratio, 'vulnerabilities.json')
        label_json = os.path.join(vuln, 'integrate', ratio, 'graph_label.json')
        with open(vuln_json, 'r') as f:
            all_vuln_items.extend(json.load(f))
        with open(clean_json, 'r') as f:
            all_clean_items.extend(json.load(f))
        with open(all_json, 'r') as f:
            all_items.extend(json.load(f))
        with open(label_json, 'r') as f:
            all_label_items.extend(json.load(f))
    
    vuln_target = os.path.join(target, ratio, 'vuln_vulnerabilities.json')
    clean_target = os.path.join(target, ratio, 'clean_vulnerabilities.json')
    all_target = os.path.join(target, ratio, 'vulnerabilities.json')
    label_target = os.path.join(target, ratio, 'graph_label.json')

    with open(vuln_target, 'w') as f:
        json.dump(all_vuln_items, f)
    with open(clean_target, 'w') as f:
        json.dump(all_clean_items, f)
    with open(all_target, 'w') as f:
        json.dump(all_items, f)
    with open(label_target, 'w') as f:
        json.dump(all_label_items, f)


if __name__ == "__main__":
    
    root_path = '/workspaces/solidity'
    dataset_dir = os.path.join(root_path, 'integrate_dataset')
    target = os.path.join(root_path, 'integrate')
    # dataset_dir = '/workspaces/solidity/integrate_dataset'
    # target = '/workspaces/solidity/integrate'

    ratio = 1
    if not os.path.exists(os.path.join(target, str(ratio))):
        os.makedirs(os.path.join(target, str(ratio)))
    integrate_all_vuln(dataset_dir, target, str(ratio))
