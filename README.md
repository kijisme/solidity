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
dataser_process.py 根据漏洞类型整合数据集smartbugs和solidifi















