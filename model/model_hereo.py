import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MetaPath2Vec

import dgl
from dgl.nn.pytorch import GATConv


from .graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, \
        get_number_of_nodes, get_node_tracker, reflect_graph, \
        get_length_2_metapath, generate_filename_ids

class SemanticAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # z [node_num, meta_path_num, input_size]
        # w [node_num, meta_path_num, 1]->[meta_path_num, 1]
        w = self.project(z).mean(0)      
        # beta [meta_path_num, 1] -> [meta_path_num, 1]      
        beta = torch.softmax(w, dim=0)
        # beta [meta_path_num, 1] -> [node_num, meta_path_num, 1] 
        beta = beta.expand((z.shape[0],) + beta.shape) 

        # [node_num, meta_path_num, input] -> [node_num, input]
        return (beta * z).sum(1)                       


class HANLayer(nn.Module):
   
    def __init__(self, meta_paths, input_size, output_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        
        self.gat_layers = nn.ModuleList()
        # 对每一条元路径添加图注意力层
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_feats=input_size, out_feats=output_size,
                                           num_heads=layer_num_heads,
                                           feat_drop=dropout, attn_drop=dropout, 
                                           activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(input_size=output_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        # h [node_num, input_size]
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()

            # 获取每条元路径的子图
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

        # 对每条元路径通过对应图注意力层后获得每层的输出
        for i, meta_path in enumerate(self.meta_paths):
            # 获取元路径子图
            new_g = self._cached_coalesced_graph[meta_path]
            # 输入对应图注意力层进行节点注意力
            # # GAT     [node_num, input_size] -> [node_num, layer_num_heads, output_size]
            # # flatten [node_num, layer_num_heads, output_size] -> [node_num, layer_num_heads*output_size]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        # 将全部输出拼接
        # h [node_num, meta_path_num, layer_num_heads*output_size]
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  

        # 全部输入语义注意力层
        # h [node_num, meta_path_num, layer_num_heads*output_size] -> [node_num, layer_num_heads*output_size]
        return self.semantic_attention(semantic_embeddings)                            

class MANDOGraphClassifier(nn.Module):
    def __init__(self, compressed_global_graph_path, feature_extractor=None, node_feature='nodetype', 
                 hidden_size=32, out_size=2, num_heads=8, dropout=0.6, device='cpu'):
        super(MANDOGraphClassifier, self).__init__()
        
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_size = hidden_size
       
        self.device = device
        
        # 获取全局图
        # 读取图文件\转化label为数字\添加'node_hetero_id'节点索引(通节点类型的第几个)
        nx_graph = load_hetero_nx_graph(self.compressed_global_graph_path)
        # 获取图的三元组 (source_type, edge_type, target_type) -> ([all_source],[all_target])
        nx_g_data = generate_hetero_graph_data(nx_graph)
        # 获取文件索引 source_file_name -> id
        self.filename_mapping = generate_filename_ids(nx_graph)
        # 获取每个类型节点涉及的文件id node_type -> [id]
        _node_tracker = get_node_tracker(nx_graph, self.filename_mapping)

        # Reflect graph data
        # 获取元路径的三元组 (source_type, edge_type, target_type) -> ([[all_source]],[[all_target]])
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        # 获取每种类型节点的数量
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        # 构图
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        # 添加所属文件索引 'filename'
        self.symmetrical_global_graph.ndata['filename'] = _node_tracker
       
        # 获取元路径
        self.length_2_meta_paths = get_length_2_metapath(self.symmetrical_global_graph)
        self.meta_paths = self.length_2_meta_paths

        # Concat the metapaths have the same begin nodetype
        # 合并具有相同起始节点类型的元路径 {ntype}:{[metapath_list]}}
        self.full_metapath = {}
        for metapath in self.meta_paths:
            ntype = metapath[0][0]
            if ntype not in self.full_metapath:
                self.full_metapath[ntype] = [metapath]
            else:
                self.full_metapath[ntype].append(metapath)

        # 全部节点类型
        self.node_types = self.symmetrical_global_graph.ntypes
        # 全部边类型
        self.edge_types = self.symmetrical_global_graph.etypes
        # 获取节点类型索引 (节点类型 ->节点索引) 
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}

        # 获取节点特征 {node_type} : {feature} 
        # feature : [node_num, in_size]
        features = self.get_node_feature(node_feature, feature_extractor, nx_graph)
        
        # 加载图到机器
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        # 设置节点特征
        self.symmetrical_global_graph.ndata['feat'] = features

        # 初始化特征提取器模型
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer([self.meta_paths[0]], self.in_size, hidden_size, num_heads, dropout))
        for meta_path in self.meta_paths[1:]:
            self.layers.append(HANLayer([meta_path], self.in_size, hidden_size, num_heads, dropout))
        # 初始化分类器MLP
        self.classify = nn.Linear(hidden_size * num_heads , out_size)

    def get_node_feature(self, node_feature, feature_extractor, nx_graph):

        features = {}
        if node_feature == 'nodetype':
            # onehot编码
            for ntype in self.symmetrical_global_graph.ntypes:
                features[ntype] = self._nodetype2onehot(ntype).repeat(self.symmetrical_global_graph.num_nodes(ntype), 1).to(self.device)
            self.in_size = len(self.node_types)
        elif node_feature == 'metapath2vec':
            embedding_dim = 128
            self.in_size = embedding_dim
            for metapath in self.meta_paths:
                # PYG
                _metapath_embedding = MetaPath2Vec(self.symmetrical_global_graph_data, 
                                                   embedding_dim=embedding_dim,
                                                   metapath=metapath, walk_length=50, context_size=7,
                                                   walks_per_node=5, num_negative_samples=5, 
                                                   num_nodes_dict=self.number_of_nodes,
                                                   sparse=False)
                ntype = metapath[0][0]
                if ntype not in features.keys():
                    features[ntype] = _metapath_embedding(ntype).unsqueeze(0)
                else:
                    features[ntype] = torch.cat((features[ntype], _metapath_embedding(ntype).unsqueeze(0)))
            features = {k: torch.mean(v, dim=0).to(self.device) for k, v in features.items()}

        return features

    def _nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature

    def get_assemble_node_features(self):
        features = {}
        for han in self.layers:
            ntype = han.meta_paths[0][0][0]
            feature = han(self.symmetrical_global_graph, self.symmetrical_global_graph.ndata['feat'][ntype])
            if ntype not in features.keys():
                features[ntype] = feature.unsqueeze(0)
            else:
                features[ntype] = torch.cat((features[ntype], feature.unsqueeze(0)))
        # Use mean for aggregate node hidden features
        return {k: torch.mean(v, dim=0) for k, v in features.items()}

    def reset_parameters(self):
        for model in self.layers:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.classify.children():
            if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, batched_g_name, save_featrues=None):
        
        # 全图输入HAN模型获得节点特征表示
        features = {}
        for han in self.layers:
            # 获取当前HAN层的节点类型
            ntype = han.meta_paths[0][0][0]
            # feature [node_num, in_size] -> [node_num, out_size]
            feature = han(self.symmetrical_global_graph, self.symmetrical_global_graph.ndata['feat'][ntype])
            if ntype not in features.keys():
                # features[ntype] [node_num, out_size] -> [meta_num, node_num, out_size]
                features[ntype] = feature.unsqueeze(0)
            else:
                features[ntype] = torch.cat((features[ntype], feature.unsqueeze(0)))
        
        # features ntype -> feature [meta_num, node_num, out_size] n为node_type出现在元路径中起始节点的次数

        batched_graph_embedded = []
        # 获取每一个文件的图嵌入
        for g_name in batched_g_name:
            # 获取文件名称的索引
            file_ids = self.filename_mapping[g_name]
            graph_embedded = 0
            for node_type in self.node_types:
                # 获取当前文件出现节点的mask
                file_mask = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
                # 保证当前文件中有图中的节点
                if file_mask.sum().item() != 0:
                    # 全部元路径中该节点的特性 [meta_path_num ,node_num, out_size] 
                    # 在第一维取平均 [meta_path_num ,node_num, out_size]->[node_num, out_size]
                    graph_embedded += features[node_type][file_mask].mean(0)
            
            # 添加到batched_graph_embedded
            batched_graph_embedded.append(graph_embedded.tolist())
        # 转化为tensor
        batched_graph_embedded = torch.tensor(batched_graph_embedded).to(self.device)
        if save_featrues:
            torch.save(batched_graph_embedded, save_featrues)
        # 输入MLP获得分类结果
        # [batch_size, node_num, out_size] ->  [batch_size, node_num, re_size]
        output = self.classify(batched_graph_embedded)

        return output, batched_graph_embedded

'''测试单个文件'''
if __name__ == '__main__':
    pass