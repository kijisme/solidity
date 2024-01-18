import paddle
from pgl.utils.data import Dataset

class contractVulnDataset(Dataset):
    def __init__(self, label_json_path):
        self._label = label_json_path

    def process(self):
        # 读取label文件
        with open(self._label, 'r') as f:
            annotations = json.load(f)
        # 获取全部图及标签
        self.graph_names, self.label = self.load_graph(annotations)

    def load_graph(self, annotations):
        graphs = []
        labels = []
        for contract in annotations:
            graphs.append(contract['contract_name'])
            labels.append(int(contract['targets']))
        labels = paddle.Tensor(labels, dtype='int64')
        return graphs, labels

    def __getitem__(self, idx):
        return self.graph_names[idx], self.label[idx]

    def __len__(self):
        return len(self.graph_names)