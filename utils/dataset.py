
import json

import torch

import dgl
from dgl.data import DGLDataset

class contractVulnDataset(DGLDataset):
    def __init__(self, label_json_path, raw_dir=None, force_reload=True, verbose=False):
        super().__init__(name='contractVuln',raw_dir=raw_dir,force_reload=force_reload,verbose=verbose)
        
        self.label_json_path = label_json_path
        # 自动执行 self.process()
        
    def process(self):
        # 获取全部label信息
        with open(self.label_json_path, 'r') as f:
            _annotations = json.load(f)
        # 获取全部图和label
        self.graphs, self.label = self._load_graph(_annotations)
        
    def _load_graph(self, _annotations):
        graphs = []
        labels = []
        for contract in _annotations:
            graphs.append(contract['contract_name'])
            labels.append(int(contract['targets']))

        labels = torch.tensor(labels, dtype=torch.int64)
        return graphs, labels

    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)