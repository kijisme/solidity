import os
import sys
import warnings
import numpy as np
from shutil import rmtree
from tabulate import tabulate
from configparser import ConfigParser

from sklearn.model_selection import KFold

import torch

from dgl.dataloading import GraphDataLoader

sys.path.append('/workspaces/solidity/utils')
from dataset import contractVulnDataset
from visualization import visualize_k_folds
from metric import score, get_classification_report, get_confusion_matrix

sys.path.append('/workspaces/solidity/model')
from model_hereo import graphClassify

warnings.filterwarnings("ignore")

def train(model, dataloader, optimizer, loss_fcn, device):
    model.train()

    metric= {'acc':0, 'macro_f1':0, 'micro_f1':0}
    total_loss = 0

    for idx, (batched_graph, labels) in enumerate(dataloader):
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(batched_graph)
        # print(logits)
        loss = loss_fcn(logits, labels)
        # 计算指标
        train_acc, train_micro_f1, train_macro_f1 = score(labels, logits)
        # 梯度下降
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
        optimizer.step()

        metric['acc'] += train_acc
        metric['micro_f1'] += train_micro_f1
        metric['macro_f1'] += train_macro_f1
        total_loss += loss.item()

    steps = idx + 1
    for item in metric:
        metric[item] = metric[item]/steps
    
    return model, total_loss/steps, metric

def validate(model, dataloader, loss_fcn, device):
    model.eval()

    total_loss = 0
    metric= {'acc':0, 'macro_f1':0, 'micro_f1':0}

    with torch.no_grad():
        for idx, (batched_graph, labels) in enumerate(dataloader):
            labels = labels.to(device)
            logits = model(batched_graph)
            loss = loss_fcn(logits, labels)
        
            val_acc, val_micro_f1, val_macro_f1 = score(labels, logits)
            
            total_loss += loss.item()
            metric['acc'] += val_acc
            metric['micro_f1'] += val_micro_f1
            metric['macro_f1'] += val_macro_f1
    
    steps = idx + 1
    for item in metric:
        metric[item] = metric[item]/steps
    
    return total_loss/steps, metric

def test(model, dataloader, device):
    model.eval()

    metric= {'acc':0, 'macro_f1':0, 'micro_f1':0}
    total_logits = []
    total_target = []
    with torch.no_grad():
        for idx, (batched_graph, labels) in enumerate(dataloader):
            labels = labels.to(device)
            logits = model(batched_graph)
            total_logits += logits.tolist()
            total_target += labels.tolist()
            test_acc, test_micro_f1, test_macro_f1 = score(labels, logits)
            
            metric['acc'] += test_acc
            metric['micro_f1'] += test_micro_f1
            metric['macro_f1'] += test_macro_f1

    steps = idx + 1
    for item in metric:
        metric[item] = metric[item]/steps

    total_logits = torch.tensor(total_logits)
    total_target = torch.tensor(total_target)

    classification_report = get_classification_report(total_target, total_logits, output_dict=True)
    confusion_report = get_confusion_matrix(total_target, total_logits)
    
    return metric, classification_report, confusion_report


def train_k_fold(dataset, total_train, kfold, batch_size, model, device, epochs, loss_fcn, optimizer):
    train_results ={}
    val_results = {}
    classification_total_report = {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                   '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 
                                   'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 
                                   'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}}
    confusion_matrix_total_report = []
    # K折训练
    for fold, (train_ids, val_ids) in enumerate(kfold.split(total_train)):
        # 划分数据集
        train_ids = [total_train[i] for i in train_ids]
        val_ids = [total_train[i] for i in val_ids]

        train_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'buggy_f1': [], 'lrs': []}
        val_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'buggy_f1': []}
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_dataloader = GraphDataLoader(dataset,batch_size=batch_size,drop_last=False,sampler=train_subsampler)
        val_dataloader = GraphDataLoader(dataset,batch_size=batch_size,drop_last=False,sampler=val_subsampler)
        print('Start training fold {} with {}/{} train/val smart contracts'.format(fold, len(train_subsampler), len(val_subsampler)))
        
        # 开始训练
        model.reset_parameters()
        model.to(device)

        for epoch in range(epochs):
            model, train_loss, train_metric = train(model, train_dataloader, optimizer, loss_fcn, device)
            val_loss, val_metric = validate(model, val_dataloader, loss_fcn, device)

            train_results[fold]['loss'].append(train_loss)
            train_results[fold]['acc'].append(train_metric['acc'])
            train_results[fold]['macro_f1'].append(train_metric['macro_f1'])
            train_results[fold]['macro_f1'].append(train_metric['macro_f1'])

            val_results[fold]['loss'].append(val_loss)
            val_results[fold]['acc'].append(val_metric['acc'])
            val_results[fold]['macro_f1'].append(val_metric['macro_f1'])
            val_results[fold]['macro_f1'].append(val_metric['macro_f1'])

        # 测试验证集
        _, classification_report, confusion_report = test(model, val_dataloader, device)
        print(classification_report)
    #     for category, metrics in classification_total_report.items():
    #         for metric in metrics.keys():
    #             classification_total_report[category][metric].append(classification_report[category][metric])

    #     confusion_matrix_total_report.append(confusion_report)
    
    # headers = ['precision', 'recall', 'f1-score', 'avg_support']
    # classification_tabular_report = []
    # for category, metrics in classification_total_report.items():
    #     row = [category]
    #     for metric in metrics.keys():
    #         std = np.std(classification_total_report[category][metric])
    #         classification_total_report[category][metric] = np.max(classification_total_report[category][metric])
    #         row.append(f'{classification_total_report[category][metric]}(#{classification_total_report[category][metric]*std:.2f})')
    #     classification_tabular_report.append(row)
    # # 打印表格数据
    # print(tabulate(classification_tabular_report, headers=headers))
    # print(np.round(np.mean(confusion_matrix_total_report, axis=0)))

    return train_results, val_results
        
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
    batch_size = int(config['train']['batch_size'])
    epochs = int(config['train']['epochs'])
    k_folds = int(config['train']['k_folds'])
    lr = float(config['train']['lr'])
    
    kfold = KFold(n_splits=k_folds, shuffle=True)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
    # lrs = []

    # 划分测试集合
    # # 设置随机种子
    np.random.seed(2)
    # # 生成随机索引
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    # # 打乱索引
    np.random.shuffle(indices)
    # # 划分训练集和测试集
    split_index = int(total_samples * (1 - test_partition))
    total_train, total_test = indices[:split_index], indices[split_index:]
    
    # 获得测试集
    test_subsampler = torch.utils.data.SubsetRandomSampler(total_test)
    test_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=test_subsampler)

    train_results, val_results = train_k_fold(dataset, total_train, kfold, batch_size, model, device, epochs, loss_fcn, optimizer)
    # # 可视化模型训练结果
    # non_visualize = int(config['train']['non_visualize'])
    # log_dir = config['train']['log_dir']
    # if non_visualize == 1:
    #     print('Visualizing')
    #     if os.path.exists(log_dir):
    #         rmtree(log_dir)
    #     visualize_k_folds(log_dir, k_folds, epochs, train_results, val_results)