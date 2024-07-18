import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

# https://www.xiaoiluo.com/article/mha#17b15507036b4ab8881dd73816b5f965

"""
d_model：也就是embedding之后的维度

d_k：在多头注意力中，参数d_k（每个头的维度）通产是由总的模型维度d_model和多头注意力的
头数决定的

当使用多个头时，每个头都在不同的子空间中进行操作，
这有助于模型捕获输入之间多样性更丰富的关系。


输入：[batch_size, seq_len, d_model]

"""

class MultiHeadAttention(nn.Module):
    def _init_(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads # 每个"头"对应的维度
        self.h = heads # "头"的数量
        
        # 初始化q k v 权重，用于生成 Q，K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 输出线性层
        self.out = nn.Linear(d_model, d_model)


    def attention(self, q, k, v, mask=None):
        # 计算点积，并通过进行缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 如果有 mask，应用于 scoresi
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 对 scores 应用 softmax
        scores = F.softmax(scores, dim=-1)

        # 应用 dropout
        scores = self.dropout(scores)
        
        # 获取输出
        output = torch.matmul(scores, v)
        
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 对 q，k，v 进行线性变换
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    
        
        # 进行多头注意力
        scores = self.attention(q, k, v, mask)

        # 将多个头的输出拼接回单个张量
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        
        # 通过输出线性层
        output = self.out(concat)

        return output 
    
if __name__ == '__main__':
    heads = 4
    d_model = 128 # d_model应该是heads的整数倍
    dropout = 0.1

    model = MultiHeadAttention(heads, d_model, dropout)
    
    # 创建模拟数据 batch_size = 2 seq_ len = 5
    batch_size = 2
    seq_len = 5

    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
    
    # 前向传播
    output = model(q, k, v)
    
    # 检查输出形状
    print("0utput shape:",output.shape) # 应该是 [batch_size, seq_len, d_model]
    
    # 检查是否可以进行反向传播
    loss = output.mean()
    loss . backward()
    
    print("Backward pass completed.")