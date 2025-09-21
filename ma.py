# class A:
#     def __new__(cls, *args,**kwargs):
#         print(cls.__name__)
#         # super(A, cls).__new__(cls)
#         self = super(A, cls).__new__(cls)
#         return self

# class B(A):
#     def __init__(self, st):
        
#         super().__init__()
#         print(f"b----------{st}")

# a = A()
# b = B
# print(type(b))
# k = B(2)

import asyncio
from typing import (Any, AsyncGenerator, Callable, Dict, Iterable, List,
                    Mapping, Optional, Set, Tuple, Type, Union)
from functools import partial
import threading

# async def foo():
#     print("foo start")
#     try:
#         await asyncio.sleep(1)
#         print(threading.current_thread().name)
#     except Exception as ex:
#         pass
# def excepta(ex: Exception):
#     print(f"Exception wangjun : {ex}")

# def print1(task: asyncio.Task,
#         error_callback: Callable[[Exception], None]):
#     try:
#         raise Exception("foo exception AAAA")
#     except Exception as ex:
#         print(threading.current_thread().name)
#         error_callback(ex)

# async def main():
#     task = asyncio.create_task(foo())  # ← 立即开始
#     task.add_done_callback(partial(print1, error_callback=excepta))
#     print("task created")
#     await asyncio.sleep(0.5)          # 让事件循环跑 0.5 秒
#     print("main continue")
#     await task                        # 等待 foo 真正结束

# asyncio.run(main())

# print(any([0, '', False]))
# l = [1,2,3]
# code = hash((*l,))
# print(code)
# A = (*l,)
# code = hash(A)
# print(code)
# print(*l)
# def func(*args, **kwargs):
#     print(args)
#     print(kwargs)

# func(*l, a=1, b=2)

# # 创建一个列表
# li = [11, 22, 33, 44, 55,66]

# # 将列表转换为迭代器
# li_iter = iter(li)

# # 使用 next() 获取下一个元素
# a = next(li_iter, None)

# while  a is not None:
#     print(a) 
#     # import pdb;pdb.set_trace()
#     a = next(li_iter, None)


# import math
# import torch
# import torch.nn.functional as F
# from torch import nn
# from transformers import GPT2Config, GPT2LMHeadModel


# class TopKRouter(nn.Module):
#     """带噪声的 Top-K 门控"""
#     def __init__(self, d_model, num_experts, top_k, aux_loss_alpha=1e-2):
#         super().__init__()
#         self.top_k = top_k
#         self.num_experts = num_experts
#         self.alpha = aux_loss_alpha
#         self.gate = nn.Linear(d_model, num_experts, bias=False)
#         self.noise = nn.Linear(d_model, num_experts, bias=False)  # 可训练噪声强度

#     def forward(self, x):  # x: [B, T, d]
#         logits = self.gate(x)                                          # [B, T, E]
#         noise = torch.randn_like(logits) * F.softplus(self.noise(x))   # 加噪声
#         noisy_logits = logits + noise

#         top_logits, indices = noisy_logits.topk(self.top_k, dim=-1)    # [B, T, K]
#         mask = torch.full_like(noisy_logits, -torch.inf)
#         mask.scatter_(-1, indices, top_logits)
#         router_probs = F.softmax(mask, dim=-1)                         # [B, T, E] 稀疏

#         # 辅助 loss：鼓励均匀路由
#         aux_loss = self.alpha * torch.mean(router_probs.sum((0, 1)) ** 2) * self.num_experts
#         return router_probs, indices, aux_loss


# class Expert(nn.Module):
#     """单个专家：两层 FFN"""
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super().__init__()
#         self.w1 = nn.Linear(d_model, d_ff)
#         self.w2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.w2(self.dropout(F.gelu(self.w1(x))))


# class MoELayer(nn.Module):
#     def __init__(self, d_model, num_experts=8, top_k=2, d_ff=None, dropout=0.1):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model
#         self.router = TopKRouter(d_model, num_experts, top_k)
#         self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout)
#                                       for _ in range(num_experts)])
#         self.top_k = top_k

#     def forward(self, x):
#         B, T, D = x.shape
#         router_probs, indices, aux_loss = self.router(x)  # [B,T,E], [B,T,K]
#         out = torch.zeros_like(x)
#         # 逐样本、逐 token 处理（教学示例，可优化为 batch 矩阵）
#         for b in range(B):
#             for t in range(T):
#                 for k in range(self.top_k):
#                     e_id = indices[b, t, k]
#                     weight = router_probs[b, t, e_id]
#                     out[b, t] += weight * self.experts[e_id](x[b, t])
#         return out, aux_loss


# class Block(nn.Module):
#     """Transformer Block：Attn + MoE"""
#     def __init__(self, d_model, n_head, **moe_kwargs):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(d_model)
#         self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.moe = MoELayer(d_model, **moe_kwargs)

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
#         x_moe, aux = self.moe(self.ln2(x))
#         x = x + x_moe
#         return x, aux


# class SparseMoELM(nn.Module):
#     def __init__(self, vocab_size, d_model=768, n_head=12, n_layer=12,
#                  num_experts=8, top_k=2, dropout=0.1):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, d_model)
#         self.pos = nn.Embedding(1024, d_model)
#         self.blocks = nn.ModuleList([
#             Block(d_model, n_head, num_experts=num_experts, top_k=top_k, dropout=dropout)
#             for _ in range(n_layer)])
#         self.ln_f = nn.LayerNorm(d_model)
#         self.head = nn.Linear(d_model, vocab_size, bias=False)

#     def forward(self, input_ids, labels=None):
#         B, T = input_ids.shape
#         pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
#         x = self.embed(input_ids) + self.pos(pos)
#         aux_loss = 0
#         for blk in self.blocks:
#             x, aux = blk(x)
#             aux_loss += aux
#         logits = self.head(self.ln_f(x))

#         if labels is None:
#             return {"logits": logits, "aux_loss": aux_loss}
#         # 计算语言模型 loss
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
#                                   shift_labels.view(-1))
#         total_loss = lm_loss + aux_loss
#         return {"loss": total_loss, "lm_loss": lm_loss, "aux_loss": aux_loss}
    

# i = [1,2,3]
# d = [1,4]

# for di in d:
#     if di in i:
#         print("d:", di)
#     for dl in d:
#         print("---------",dl)
#     else:
#         print("i:", di) 
import secrets

salt  = secrets.token_bytes(16)

print(salt.decode('GBK', 'ignore'))

print(__name__, __file__)