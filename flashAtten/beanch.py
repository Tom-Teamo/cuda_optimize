import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flashAtten_v1.cu', 'flashAtten_v2.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

torch_mHead = nn.MultiheadAttention(embed_dim=head_embd*n_head, num_heads=n_head, batch_first=True).to('cuda')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    # q = q.view(q.shape[0], seq_len, -1)
    # k = k.view(q.shape[0], seq_len, -1)
    # v = v.view(q.shape[0], seq_len, -1)
    # y, _ = torch_mHead(q, k, v)
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# manual_result = manual_result.reshape(batch_size, n_head, seq_len, head_embd)

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result_v2 = minimal_attn.forward_v2(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result_v2, manual_result, rtol=0, atol=1e-02))
