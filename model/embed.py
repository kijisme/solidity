import os
import pickle
import numpy as np

from gensim.models import Word2Vec
# import nltk
from nltk.tokenize import word_tokenize


from graph_utils import load_hetero_nx_graph

def get_content_embedding(content_text, key_num, embedding_dim, window, min_count, workers):
    
    # 分词
    tokenized_texts = [word_tokenize(text.lower()) for text in content_text if text is not None]
    # 训练word2vec模型
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, window=window, min_count=min_count, workers=workers)
    # 获取content嵌入
    content_emb = []
    for text in content_text:
        if text is None:
            emb = np.zeros(shape=(key_num, embedding_dim))
        else:
            emb = word2vec_model.wv[word_tokenize(text.lower())]
            if emb.shape[0] >= key_num:
                emb = emb[:key_num]
            else:
                padding = np.zeros(shape=(key_num-emb.shape[0], embedding_dim))
                emb = np.concatenate([emb, padding])
        content_emb.append(emb)
    # content_emb = [word2vec_model.wv[word_tokenize(text.lower())] for text in content_text]
    
    return content_emb

'''测试单个文件'''
if __name__ == '__main__':
    # 下载语料库
    # nltk.download('punkt')
    compressed_global_graph_path = '/workspaces/solidity/integrate_dataset/unchecked_low_level_calls/integrate/1/compress.gpickle'
    # 读取图文件
    nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
    node_content = [node_data['node_expression'] for _, node_data in nx_graph.nodes(data=True)]
    key_num = 15
    embedding_dim = 16
    # 对content进行编码
    content_emb = get_content_embedding(node_content, key_num, embedding_dim, window=5, min_count=1, workers=4)
    # 保存编码信息
    save_path = os.path.join(compressed_global_graph_path.split('compress.gpickle')[0], 'content_emb.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(content_emb, file)