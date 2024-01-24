
import torch
import torch.nn as nn
import torch.nn.functional as F

class fuse_layer(nn.Module):
    def __init__(self, in_feats_a, in_feat_c, out_feat, num_heads):
        super().__init__()
        self.in_feat_a = in_feats_a
        self.in_feat_c = in_feat_c
        self.out_feat = out_feat

        self.linear_a = nn.Linear(self.in_feat_a, self.out_feat,dtype=torch.float64)
        self.lstm_c = nn.LSTM(self.in_feat_c, int(self.out_feat/2), 2, bidirectional = True ,dtype=torch.float64)

        self.atten = nn.MultiheadAttention(self.out_feat, num_heads, dtype=torch.float64)

    def forward(self, h_a, h_c):
        # h_a [node_num, input_size_1] 
        # h_c [node_num, seq, input_size_2]
        print(h_a.shape, h_c.shape)
        # a_out [node_num, out_size]
        a_out = self.linear_a(h_a)
        # a_out [node_num, 1, out_size]
        a_out = a_out.unsqueeze(1)
        # c_out [node_num, seq, out_size]
        c_out, (_, _) = self.lstm_c(h_c)
        in_att = torch.concatenate([a_out, c_out], axis=1)
        # [node_num, seq+1, out_size] -> [node_num, seq+1, out_size]
        out , _= self.atten(in_att, in_att, in_att)
        
        # [node_num, out_size]
        out = out.mean(1)

        return out

class fuse(nn.Module):
    def __init__(self, in_feats_a, in_feat_c, out_feat, num_heads):
        super().__init__()
        self.layers = fuse_layer(in_feats_a, in_feat_c, out_feat, num_heads)
       
    def forward(self, h_a, h_c):

        features_a = []
        features_c = []

        for ntype in h_a:
            features_a.append(h_a[ntype])
        for ntype in h_c:
            features_c.append(h_c[ntype])
        
        features_a = torch.vstack(features_a)
        features_c = torch.vstack(features_c)
        # [node_num, out_size]
        features = self.layers(features_a, features_c)

        features_r = {}
        for ntype in h_a:
            features_r[ntype] = features[]
            
        return {k: v for k, v in features.items()}

'''测试单个文件'''
if __name__ == '__main__':

    in_feats_a = 10
    in_feat_c = 128
    out_feat = 32
    num_heads = 4

    layer = fuse_layer(in_feats_a, in_feat_c, out_feat, num_heads)
    input_a = torch.randn(size=(3,in_feats_a))
    input_c = torch.randn(size=(3, 3 ,in_feat_c))
    out = layer(input_a, input_c)
    print(out.shape)
