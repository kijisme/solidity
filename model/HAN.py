import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv


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

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.meta_paths = meta_paths
        self.in_size = in_size
        # 初始化特征提取器模型
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer([self.meta_paths[0]], self.in_size, hidden_size, num_heads, dropout))
        for meta_path in self.meta_paths[1:]:
            self.layers.append(HANLayer([meta_path], self.in_size, hidden_size, num_heads, dropout))

    def forward(self, g, h):
        features = {}
        for han in self.layers:
            ntype = han.meta_paths[0][0][0]
            # print(h[ntype])
            feature = han(g, h[ntype])
            if ntype not in features.keys():
                features[ntype] = feature.unsqueeze(0)
            else:
                features[ntype] = torch.cat((features[ntype], feature.unsqueeze(0)))
      
        return {k: torch.mean(v, dim=0) for k, v in features.items()}
