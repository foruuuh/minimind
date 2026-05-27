from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )





import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add(self.eps))
        return x
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).tpye_as(x) * x
    

def precompute_freqs_cis(dim:int, end:int(32*1024), rope_base, rope_scaling:Optional[dict] = None):
    # 初始化RoPE频率
    freqs,attn_factor = (1.0/(rope_base**(torch.arange(0, dim, 2).float()/dim)), 1.0)

    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"]
        )

        # 推断的长度大于训练长度，使用缩放
        if end > orig_max:
            # 波长b到i的映射
            inv_dim = lambda b:(dim * math.log(orig_max / (b * 2 * math.pi)))/(2 * math.log(rope_base))
            # 划分高低维度
            #low： 不需要缩放的维度，high：需要缩放的维度
            low, high = (max(math.floor(inv_dim(beta_fast))), min(math.ceil(inv_dim(beta_slow)), dim // 2-1))

            # 计算缩放因子
            # low之前， ramp为0，在high之后，ramp为1，在low和high之间，先行过度
            ramp = torch.clamp(
                (torch.arange(dim // 2, device = freqs.device).float() - low) / max(high - low, 0.001),

                0,
                1,
            )

            reqs = freqs * (1 - ramp + ramp / factor)
        # 根据end，生成位置索引t
        t = torch.arange(end, device = freqs.device).float()

        # 计算外积，将t和频率部分相乘，得到每个位置的旋转角度
        freqs = torch.outer(t, freqs).float()

        freqs_cos = (
            torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor
        )
        freqs_sin = (
            torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor
        )

        return freqs_cos, freqs_sin
            
#  编写RoPE
def apply_rotarry_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    # [a, b] -> [-b, a]
    def rotate_half(x):
        # x.shape[-1]取最后一个维度的重点
        # x[..., : x.shape[-1] // 2]取前半部分，x[..., x.shape[-1] // 2 :]取后半部分
        return torch.cat([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim = -1)
    
    # x__rotated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed


def repeat_kv(x:torch.tensor, n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        )

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is None else args.num_attention_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0,
        "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.out_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias = False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention


        def forward(
            self,
            x:torch.Tensor,
            position_embedding:Tuple[torch.Tensor, torch.Tensor], 
            past_cache:Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
            use_cache = False,
            attention_mask:Optional[torch.Tensor] = None,
        )-> torch.Tensor:
            # 投影，计算qkv
            bsz,seq_len,_ = x.shape
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            # 把输入拆成多个头，用view
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            # qk，使用RoPE
            cos, sin = position_embedding
            xq, xk = apply_rotarry_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
            # 对于kv，使用repeat（注意kv cache）
            if past_key_value is not None:
                xk = torch.cat([past_cache[0], xk], dim = 1)
                xv = torch.cat([past_cache[1], xv], dim = 1)
            past_kv = (xk, xv) if use_cache else None

            xq, xk, xv = (
                xq.transpose(1, 2),
                # [bsz, n_local_heads, seq_len, head_dim]
                repeat_kv(xk, self.n_rep).transpose(1, 2),
                repeat_kv(xv, self.n_rep).transpose(1, 2),
            )
            # 进行attention计算，QK^T / sqrt(d)
            if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask==1)):
                attn_mask=(
                    None
                    if attention_mask is None
                    else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
                )
                output = F.scaled_dot_product_attention(xq, xk ,xv, attn_mask, dropout_p = self.dropout if self.training else 0.0, is_causal = True)
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores = scores + torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device = scores.device), diagonal = 1
                ).unsqueeze(0).unsqueeze(0)

                if attention_mask is not None:
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                    scores = scores + extended_attention_mask

                scores = F.softmax(scores.float(), dim = -1).type_as(xq)
                scores = self.attn_dropout(scores)
                output = scores @ xv

            output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
            output = self.resid_dropout(self.o_proj(output))
            return output, past_kv

