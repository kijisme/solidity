import os
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dglnn

from fuse import fuse
from graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, reflect_graph, \
                         get_number_of_nodes, get_symmatrical_metapaths, \
                         get_node_tracker, get_length_2_metapath, map_node_embedding

from HAN import HAN

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class graphClassify(nn.Module):
    def __init__(self, compressed_global_graph_path, source_path, content_emb, 
                 in_size, hidden_size=32, out_size=2, num_heads=8, dropout=0.6, device='cpu'):
        super().__init__()

        self.source_path = source_path
        self.device = device
        with open(source_path, 'r') as f:
            self.extracted_graph = [item['contract_name'] for item in json.load(f)]
        # self.extracted_graph = [f for f in os.listdir(self.source_path) if f.endswith('.sol')]
        self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}
        # print(self.filename_mapping)
        
        # 加载全局图
        self.hidden_size = hidden_size
        self.in_size = in_size
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        _node_tracker = get_node_tracker(nx_graph, self.filename_mapping)
        
        # 转化为异构图
        nx_g_data = generate_hetero_graph_data(nx_graph)
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        self.symmetrical_global_graph.ndata['filename'] = _node_tracker

        # 获取图结构信息
        self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
        self.meta_paths_2 = get_length_2_metapath(self.symmetrical_global_graph)
        # self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
        # self.edge_types = set([meta_path[0][1] for meta_path in self.meta_paths])
        self.node_types = self.symmetrical_global_graph.ntypes
        self.edge_types = self.symmetrical_global_graph.etypes
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}

        # 获取图特征信息
        features_attribute, features_content = self.get_node_feature(nx_graph, content_emb)

        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        self.symmetrical_global_graph.ndata['attribute'] = features_attribute
        self.symmetrical_global_graph.ndata['content'] = features_content
        # self.symmetrical_global_graph.ndata['feat'] = features

        # 初始化模型 in_feats_a, in_feat_c, out_feat, num_heads
        self.fuse = fuse(self.node_types, self.attribute_size, self.emb_size, self.in_size, num_heads)
        # # RGCN
        # self.rgcn = RGCN(self.in_size, self.hidden_size, self.hidden_size, self.edge_types)
        # self.classify = nn.Linear(self.hidden_size, out_size)
        # # HAN
        self.han = HAN(self.meta_paths_2, self.in_size, self.hidden_size, num_heads=num_heads, dropout=dropout)
        
        # self.classify = nn.Linear(self.hidden_size*num_heads, out_size)
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_size*num_heads, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Softmax(dim=1)
        )

    def get_node_feature(self, nx_graph, content_emb):
        features_attribute = {}
        features_cotent = {}
        
        # 节点属性的编码
        for ntype in self.symmetrical_global_graph.ntypes:
            features_attribute[ntype] = self._nodetype2onehot(ntype).repeat(self.symmetrical_global_graph.num_nodes(ntype), 1).to(self.device)
        
        # 节点特征的编码
        with open(content_emb, 'rb') as f:
            embedding = pickle.load(f, encoding="utf8")
        
        embedding = np.array(embedding)
        embedding = torch.tensor(embedding, dtype=torch.float, device=self.device)
        features_cotent = map_node_embedding(nx_graph, embedding)

        self.attribute_size = len(self.ntypes_dict)
        self.emb_size = embedding.shape[-1]

        return features_attribute, features_cotent

    def _nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature
    
    def reset_parameters(self):
        
        for model in self.classify:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        for layer in self.han.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.fuse.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, batched_graph):
        batch_output = []
        for g_name in batched_graph:
            file_ids = self.filename_mapping[g_name]
            node_mask = {}
            for node_type in self.node_types:
                mask = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
                if mask.sum(0) != 0:
                    node_mask[node_type] = mask 
            # 获取子图（相同的结构数据）
            sub_graph = dgl.node_subgraph(self.symmetrical_global_graph, node_mask)
            # 特征聚合
            attribute_emb = sub_graph.ndata['attribute']
            content_emb = sub_graph.ndata['content']
            # [node_num, in_size]
            # print(attribute_emb.dtype, content_emb.dtype)
            fuse_emb = self.fuse(attribute_emb, content_emb)
            sub_graph.ndata['feat'] = fuse_emb
            # sub_graph.ndata['feat'] = attribute_emb
            features =  self.han(sub_graph, sub_graph.ndata['feat'])
            sub_graph.ndata['h'] = features
            # 聚合特征
            hg = []
            for _, feature in sub_graph.ndata['h'].items():
                hg.append(feature)
            hg = torch.vstack(hg).sum(0)
            batch_output.append(hg)
        batch_output = torch.vstack(batch_output)
        output = self.classify(batch_output)

        return output

    # def forward(self, batched_graph):

    #     batch_output = []
    #     features =  self.han(self.symmetrical_global_graph, self.symmetrical_global_graph.ndata['feat'])
    #     self.symmetrical_global_graph.ndata['h'] = features

    #     for g_name in batched_graph:
    #         file_ids = self.filename_mapping[g_name]
    #         # print(file_ids)
    #         node_mask = {}
    #         for node_type in self.node_types:
    #             mask = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
    #             if mask.sum(0) != 0:
    #                 node_mask[node_type] = mask 
    #         # 获取子图（相同的结构数据）
    #         sub_graph = dgl.node_subgraph(self.symmetrical_global_graph, node_mask)

    #         hg = []
    #         for _, feature in sub_graph.ndata['h'].items():
    #             hg.append(feature)
    #         hg = torch.vstack(hg).sum(0)
    #         batch_output.append(hg)

    #     batch_output = torch.vstack(batch_output)
    #     output = self.classify(batch_output)

    #     return output
            


    # def forward(self, batched_graph):
    #     batch_output = []
    #     for g_name in batched_graph:
    #         # print(g_name)
    #         file_ids = self.filename_mapping[g_name]
    #         # print(file_ids)
    #         node_mask = {}
    #         for node_type in self.node_types:
    #             mask = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
    #             if mask.sum(0) != 0:
    #                 node_mask[node_type] = mask 
            
    #         # 获取子图（相同的结构数据）
    #         sub_graph = dgl.node_subgraph(self.symmetrical_global_graph, node_mask)
    #         # 获取子图特征数据
    #         h = sub_graph.ndata['feat']
    #         # 特征提取
    #         output_graph = self.rgcn(sub_graph, h)
    #         # 输入到子图
    #         # print(output_graph.keys())
    #         sub_graph.ndata['h'] = output_graph

    #         # 获取图特征
    #         hg = []
    #         for ntype, feature in sub_graph.ndata['h'].items():
    #             hg.append(feature)
    #         hg = torch.vstack(hg).sum(0)
    #         batch_output.append(hg)

    #     # 分类
    #     batch_output = torch.vstack(batch_output)
    #     output = self.classify(batch_output)
    #     return output
       

'''测试单个文件'''
if __name__ == '__main__':
    compress_graph = '/workspaces/solidity/integrate_dataset/other/integrate/compress.gpickle'
    source_path = '/workspaces/solidity/integrate_dataset/other/integrate/graph_label.json'
    content_emb = '/workspaces/solidity/integrate_dataset/other/integrate/content_emb.pkl'
    in_size = 32
    classfier = graphClassify(compress_graph, source_path, content_emb, in_size)

    example_graph = ['smartbugs_other_crypto_roulette.sol', 'clean_0x921ae917e843a956650f2bddd95446188cf08b38.sol']

    output = classfier(example_graph)
    print('output', output.shape, torch.isnan(output).any().item())
    input = torch.randn(size=(output.shape), requires_grad=True)
    loss = F.cross_entropy(input, output)
    print(output)
    loss.backward()
    
    # # 查看梯度
    for name, param in classfier.named_parameters():
        if param.grad is None:
            print(name)