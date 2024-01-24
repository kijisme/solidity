import os
import sys
from configparser import ConfigParser

from sklearn.model_selection import KFold

import torch

sys.path.append('/workspaces/solidity/utils')
from dataset import contractVulnDataset

sys.path.append('/workspaces/solidity/model')
from model_hereo import graphClassify

def train(dataset, test_partition, kfold, model, loss_fcn, optimizer):
    # 划分数据集

    pass

'''测试单个文件'''
if __name__ == '__main__':
    # 读取配置信息
    config_file = '/workspaces/solidity/config.ini'
    config = ConfigParser()
    config.read(config_file, encoding='UTF-8')

    # 读取文件信息
    dataset_dir = config['file']['dataset_dir']
    label_json_path = os.path.join(dataset_dir, 'graph_label.json')
    compressed_global_graph_path = os.path.join(dataset_dir, 'compress.gpickle')
    content_emb = os.path.join(dataset_dir, 'content_emb.pkl')
    # 获取数据集
    dataset = contractVulnDataset(label_json_path)

    # 读取模型信息
    in_size = int(config['model']['in_size'])
    hidden_size = int(config['model']['hidden_size'])
    out_size = int(config['model']['out_size'])
    num_heads = int(config['model']['num_heads'])
    dropout = float(config['model']['dropout'])
    device = config['train']['device']
    # 获取模型
    model = graphClassify(compressed_global_graph_path, label_json_path, content_emb, 
                 in_size, hidden_size, out_size, num_heads, dropout, device)
    
    # 获取训练信息
    test_partition = float(config['train']['test_partition'])
    k_folds = config['train']['k_folds']
    lr = float(config['train']['lr'])
    
    kfold = KFold(n_splits=k_folds, shuffle=True)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
    # lrs = []

    train(dataset, test_partition, kfold, model, loss_fcn, optimizer)



