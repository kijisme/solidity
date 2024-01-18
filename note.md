# 已解决
1. cfg空点 (function节点的id重复出现)
2. cg中存在cfg中不存在的点 (不存在的点观察有:Variable)
3. cfg中stateVariable的信息填写
4. dataset_collect.py 随机选取clean数据重复
# 待解决
1. 使用pgl or dgl
2. 模型为metapath2vec or node2vec
3. HAN的处理步骤
# 考虑
1. 是否将全局变量的调用考虑到cg中 ?
yes : cg多余的点均为stateVariable ? 
    yes : combineGaph需要提取两种节点(function和stateVariable) 
    no : 全部变量都需要考虑




# 对于合约继承处理: 
* 注意合约不访问private成员,但可以通过继承合约的函数来访问private变量,但private仍属于父合约
slither将继承的外部函数加入到当前合约中,将继承获得的节点中的合约相关信息,使用当前合约
# 对于外部调用: 
对于外部调用的函数,cfg节点只考虑顺序和控制语句,每个语句内部只考虑使用到的localvariable和statevariable,其中使用到的函数,由cg外部调用表示
对于外部调用的全局变量,使用getter方法获得,视为外部调用函数,由cfg的数据流-->cg的外部调用的variable关系
# 对于内部调用:
对于内部调用函数,使用cg内部调用表示
对于内部调用变量,使用cfg数据流表示

# 总:在cfg中,只考虑合约内部数据流动和语句控制关系,合约内部函数关系和合约的外部调用由cf考虑

# 对于图合并:
func_graph -> contract_graph      
---需要考虑状态变量，不同函数需要访问统一状态变量，**函数间不独立**
---func_graph的id可能存在重复,relabel:f'{function.full_name}_id'使用nx.compose(),
---添加内部状态变量数据边,(f'{function.name}_id', 状态变量名称)
contract_graph -> sol_file_graph
---合约继承关系，文件内存在不同合约具有相同函数，故：id：合约名称\函数名称|状态变量名称
---合约继承时，对于使用public继承函数去访问父合约private合约状态变量的情况，合约函数中语句与另一合约状态变量的使用关系，**合约间不独立**
---使用nx.compose(),避免重复的跨合约状态变量
sol_file_graph -> bug_graph
---一个.sol为一个编译单元，**文件间独立**
---使用nx.disjoint_union()

# 对于节点id & token信息
状态变量 & 函数节点: 合约内部唯一,文件内部和文件间可能存在重复,cg是在文件粒度上进行处理
---id : 合约名称\节点名称
---token:文件名称\合约名称\节点名称
* 对于函数节点,智能合约支持重载,节点名称为full_name
控制流节点:函数内唯一,合约内部,文件内部和文件间可能存在重复,在cfg基础上处理
----token:node_type\node_expression

# token
[状态变量]:
str(sol_file_name)
str(contract.name),
str(state_var.full_name)
[控制流节点]:
str(contract.name),
str(function.name),
str(node.node_id)
[函数节点]:
str(sol_file_name)
str(contract.name)
str(function.full_name)

[控制流] [顺序流] 控制流节点内部关系
[数据流] 
--状态流数据边 控制流节点与控制流节点的关系
--局部数据边 控制流节点与状态变量的关系
[外部调用] 合约内部调用
[外部调用] 合约外部调用 -- 状态变量使用getter函数获取,指向合约的状态变量