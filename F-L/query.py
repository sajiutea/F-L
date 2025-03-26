import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd

data = pd.read_csv("/root/autodl-fs/query.csv")
X=data['definition']
X = X.tolist()
print(X)

# 加载预训练的BERT模型和分词器
bert_name = '/root/autodl-fs/CXR-BERT_specialized'
model = AutoModel.from_pretrained(bert_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(bert_name,trust_remote_code=True)


inputs = tokenizer(X, return_tensors='pt', padding='max_length', truncation=True,max_length=128)
outputs = model(**inputs)
# 使用CLS token的嵌入作为查询向量
cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 获取CLS token嵌入
# 定义降维层，将768维降到512维
reduction_layer = nn.Linear(768, 512)
# 进行降维
query_embeddings = reduction_layer(cls_embeddings)  # 维度转换为 [batch_size, 512]
# 将查询向量保存到文件中
torch.save(query_embeddings, '/root/autodl-fs/query_embeddings.pt')
print("Query embeddings saved to 'query_embeddings.pt'")

# 从文件中加载查询向量
query_embeddings = torch.load('/root/autodl-fs/query_embeddings.pt')

# 转换为torch.nn.Parameter并设置requires_grad=False
query_embeddings = nn.Parameter(query_embeddings, requires_grad=False)

# 打印查询向量以检查
print(query_embeddings)


