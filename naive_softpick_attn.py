import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

def softpick(x, dim=-1, eps=1e-8):
    # softpick function: relu(exp(x)-1) / sum(abs(exp(x)-1))
    # numerically stable version
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps) # epsilon is only useful if all inputs are EXACTLY 0. we might not even need it

def naive_softpick_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    if not head_first:
        q, k, v = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (q, k, v))
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    mask = torch.tril(torch.ones(k_len, k_len, device=q.device))
    wei = torch.matmul(q, k.transpose(2, 3)) # shape: (batch_size, num_heads, q_len, k_len)
    wei = wei * scale
    wei = wei.masked_fill(mask[k_len-q_len:k_len, :k_len] == 0, float('-inf'))
    wei = softpick(wei.float(), dim=-1).to(q.dtype)
    o = torch.matmul(wei, v) # shape: (batch_size, num_heads, q_len, head_dim)
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')
    return o, wei
