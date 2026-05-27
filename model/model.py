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
from transformers.activations import ACT2FN
from transformers import PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

    
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

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

        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0,
        "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
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
            position_embeddings:Tuple[torch.Tensor, torch.Tensor], 
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
            cos, sin = position_embeddings
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
            if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask == 1)):
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
        


class FeedForward(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 63) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias = False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias = False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias = False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))
        )
    

class MokioMindBlock(nn.Module):
    def __init__(self, layer_id:int,config:MokioMindConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForward(config)
    

    def forward(self,
                hidden_states, 
                position_embedding, 
                past_key_value = None,
                use_cache = False,
                attention_mask = None,
                ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embedding,
            past_cache = past_key_value,
            use_cache = use_cache,
            attention_mask = attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    
class MokioMindModel(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()
        
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

        # RoPE预计算
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )

        self.register_buffer("freqs_cos", self.freqs_cos, persistent = False)
        self.register_buffer("freqs_sin", self.freqs_sin, persistent = False)

    def forward(
            self,
            input_ids:Optional[torch.Tensor] = None,
            attention_mask:Optional[torch.Tensor] = None,
            past_key_values:Optional[Tuple[torch.Tensor]] = None,
            use_cache:bool = False,
            **kwargs, 
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embedding = (self.freqs_cos[start_pos : start_pos + seq_len], 
                              self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embedding,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask = attention_mask,
            )

            presents.append(present)

        hidden_states = self.norm(hidden_states)
         
        return hidden_states, presents
    

class MokioMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config:MokioMindConfig):
        self.config = config

        super().__init__(config)

        self.model = MokioMindModel(config)

        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False)


        # 权重共享
        # 输出层权重和嵌入层的权重共享
        self.model.embed_tokens.weight = self.lm_head.weight


    def forward(self, input_ids:Optional[torch.Tensor] = None,
                attention_mask:Optional[torch.Tensor] = None,
                past_key_values:Optional[Tuple[Tuple[torch.Tensor]]] = None,
                use_cache:bool = False,
                Logits_to_keep:Union[int, torch.Tensor] = 0,
                **args,
    ):
        hidden_states, past_key_values = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache,
            **args,
        )
        slice_indices = (slice(-Logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits = logits,
            past_key_values = past_key_values,
            hidden_states = hidden_states,
        )