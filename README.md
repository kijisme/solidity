# solidity


0.5.11 0.5.10 0.5.9 0.5.8 0.5.7 0.5.6 0.5.5 0.5.4 0.5.3 0.5.2 0.5.1 0.5.0

0.4.9 0.4.8 0.4.6 0.4.4 0.4.3 0.4.2 0.4.1 0.4.0 [5,8]

0.4.19 0.4.18 0.4.17 0.4.16 0.4.15 0.4.14 0.4.13 0.4.12 0.4.11 0.4.10

0.4.26 0.4.25 0.4.24 0.4.23 0.4.22 0.4.21 0.4.20

/workspaces/solidity/json/clean.json 全部的clean数据集  
/workspaces/solidity/json/error.json clean数据集中解析不出来的  
/workspaces/solidity/json/smartbugs.json smartbugs数据集全部的数据  
/workspaces/solidity/json/solidifi.json solidifi数据集中全部的数据  


solc_install.py 获取代码中指出的solc版本并保存到json/{dataset}.json中  
solc_process.py 根据json中指出的solc版本对代码进行slither解析，对于clean错误的直接删除该项，对于smartbugs和solidifi修改正确版本  
solc_test.py 测试json文件中version解析成果  

dataset_process.py 根据漏洞类型整合数据集smartbugs和solidifi  
dataset_collect.py 为每种漏洞选择clean,数据合并json  

-ld ./logs/graph_classification/cfg_cg/node2vec/access_control 
--output_models ./models/graph_classification/cfg_cg/node2vec/access_control 
--dataset ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/ 
--compressed_graph ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_cg_compressed_graphs.gpickle 
--label ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/graph_labels.json --node_feature node2vec 
--feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_access_control_cfg_cg_clean_57_0.pkl 
--seed 1

节点属性特征没有考虑:
id,
√ node_type,
√ node_expression,
× node_token,
× node_code_lines, 占位
× node_vuln_info,
 function_fullname,
× contract_name, 
× source_file













