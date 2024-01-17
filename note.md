# 已解决
1. cfg空点 (function节点的id重复出现)
2. cg中存在cfg中不存在的点 (不存在的点观察有:Variable)
3. cfg中stateVariable的信息填写
# 待解决
1. 使用pgl or dgl
2. 模型为metapath2vec or node2vec
3. HAN的处理步骤
# 考虑
1. 是否将全局变量的调用考虑到cg中 ?
yes : cg多余的点均为stateVariable ? 
    yes : combineGaph需要提取两种节点(function和stateVariable) 
    no : 全部变量都需要考虑

合约继承:
    状态变量:
        加入节点时: 名称 -> 当前合约
        连接边时: 名称 -> 声明合约