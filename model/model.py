import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMSNorm归一化层
    相比LayerNorm，RMSNorm只进行缩放而不进行平移，计算效率更高
    """
    def __init__(self, dim: int, eps: float):
        """
        初始化RMSNorm层
        
        参数:
            dim: 输入特征维度
            eps: 数值稳定性的小常数
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            归一化后的张量
        """
        # 计算RMS(均方根)并归一化，然后应用缩放参数
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算旋转位置编码(RoPE)的复数表示
    
    RoPE的数学原理：
    - 对于位置m和维度d，RoPE定义了旋转矩阵：R(θ_m,d) = [cos(m·θ_d), -sin(m·θ_d); sin(m·θ_d), cos(m·θ_d)]
    - 其中θ_d = theta^(-2d/dim)，dim是总维度
    - 使用复数表示可以将旋转简化为复数乘法：e^(i·m·θ_d) = cos(m·θ_d) + i·sin(m·θ_d)
    
    RoPE的相对位置编码特性：
    - 虽然RoPE为每个绝对位置分配固定的编码向量，但其设计使得在注意力计算中自然地体现相对位置关系
    - 当位置m的查询向量(q·e^(im·θ))与位置n的键向量(k·e^(in·θ))计算注意力分数时
    - 结果为q·k·e^(i(m-n)·θ)，只依赖于相对位置(m-n)，而不依赖于绝对位置
    
    参数:
        dim: 注意力头的维度(head_dim)，而非模型的总嵌入维度
             在多头注意力中，每个头处理的是原始嵌入的一个独立投影，维度为head_dim
        end: 预计算的最大序列长度，默认32K
        theta: 频率衰减参数，控制不同维度的旋转频率
        
    返回:
        形状为[end, dim/2]的复数张量，表示每个位置在每个维度上的旋转因子
    """
    # 步骤1: 计算不同维度的频率
    # - 创建一个从0开始，步长为2的序列，长度为dim//2
    # - 将这个序列除以dim，得到一个从0到接近1的均匀分布值
    # - 将theta的这些值次方作为分母
    # - 结果是一组从1递减的频率值，高维度对应低频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 步骤2: 创建位置索引，从0到end-1
    t = torch.arange(end, device=freqs.device)  # 位置索引
    
    # 步骤3: 计算位置-频率矩阵（外积）
    # - 结果是一个形状为[end, dim//2]的矩阵
    # - 每行对应一个位置，每列对应一个频率维度
    # - 矩阵中的每个元素表示：位置m在频率d上的旋转角度
    freqs = torch.outer(t, freqs).float()  # 计算外积得到位置-频率矩阵
    
    # 步骤4: 转换为复数形式
    # - 使用torch.polar将频率转换为复数形式
    # - 幅值为1，相位为freqs
    # - 等价于e^(i·freqs)，即欧拉公式
    # - 这个复数表示了旋转变换
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转换为复数形式
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """
    应用旋转位置编码到查询和键向量
    
    RoPE的应用原理：
    - 将查询和键向量视为复数，然后与位置编码进行复数乘法
    - 复数乘法实现了旋转变换，保持向量长度不变，只改变方向
    - 这种旋转使得模型能够感知token的绝对位置和相对位置关系
    
    相对位置编码的实现：
    - 当位置m的查询向量(q·e^(im·θ))与位置n的键向量(k·e^(in·θ))计算注意力分数时
    - 结果为q·k·e^(i(m-n)·θ)，只依赖于相对位置(m-n)
    - 这种设计使得模型在不需要显式计算相对位置的情况下，自然地获得相对位置信息
    
    参数:
        xq: 查询向量 [batch_size, seq_len, n_heads, head_dim]
            这是原始嵌入经过线性投影并重塑后的多头表示，每个头处理一个独立的投影
        xk: 键向量 [batch_size, seq_len, n_heads, head_dim]
            同上
        pos_cis: 预计算的位置编码 [seq_len, head_dim/2]（复数形式）
            维度是head_dim/2是因为在复数表示中，每个复数占用两个实数值
        
    返回:
        应用位置编码后的查询和键向量
    """
    def unite_shape(pos_cis, x):
        """
        调整pos_cis的形状以匹配输入张量，便于广播运算
        
        参数:
            pos_cis: 位置编码 [seq_len, head_dim/2]
            x: 输入张量 [batch_size, seq_len, n_heads, head_dim/2]（复数形式）
            
        返回:
            调整形状后的位置编码
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        # 创建新形状，只保留seq_len和head_dim维度，其他维度设为1
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 步骤1: 将向量转换为复数形式
    # - 将最后一个维度每两个元素组合成一个复数
    # - 例如，如果原始形状是[batch, seq_len, n_heads, head_dim]
    # - 转换后形状变为[batch, seq_len, n_heads, head_dim//2]，但每个元素是复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 步骤2: 调整位置编码形状，使其能与输入张量正确广播
    pos_cis = unite_shape(pos_cis, xq_)
    
    # 步骤3: 应用旋转（复数乘法）
    # - 将复数形式的查询和键向量与位置编码相乘
    # - 复数乘法实现了旋转变换
    # - 这一步使得每个位置的向量都乘以对应位置的旋转因子
    # - 在注意力计算中，这些旋转后的向量会自然地体现出相对位置关系
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    
    # 步骤4: 转换回原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值向量以匹配多头注意力中的头数
    
    参数:
        x: 键或值向量
        n_rep: 重复次数
        
    返回:
        重复后的张量
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在新维度上扩展并重塑
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制实现，支持分组查询注意力(GQA)
    """
    def __init__(self, args: LMConfig):
        """
        初始化注意力层
        
        参数:
            args: 模型配置
        """
        super().__init__()
        # 设置键值头数，如果未指定则等于查询头数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads  # 查询头数
        self.n_local_kv_heads = self.n_kv_heads  # 键值头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个键值头对应的查询头数
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        
        # 定义查询、键、值的线性投影
        # 这些投影将输入从dim维投影到(n_heads*head_dim)维，然后重塑为多头形式
        # 多头注意力不是简单地分割输入维度，而是通过不同的投影矩阵创建不同的表示
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出投影
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # 检查是否使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        
        # 创建因果掩码（上三角矩阵）
        # 创建一个形状为(1,1,max_seq_len,max_seq_len)的掩码张量，填充负无穷
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        # 将掩码转换为上三角矩阵，对角线以上的元素保持为负无穷，实现因果注意力机制
        mask = torch.triu(mask, diagonal=1)
        # 将掩码注册为模型的缓冲区，但不作为持久状态保存（persistent=False）
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, dim]
            pos_cis: 位置编码，用于RoPE
            past_key_value: KV缓存，用于推理加速
            use_cache: 是否使用KV缓存
            
        返回:
            输出张量和更新的KV缓存
        """
        bsz, seq_len, _ = x.shape
        
        # 线性投影得到查询、键、值
        # 这一步将输入x投影到查询、键、值空间，输出维度是n_heads*head_dim
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 重塑为多头形式
        # 这一步将连续的投影结果重塑为多头形式，每个头获得一个head_dim维的向量
        # 这些向量是原始输入的不同投影表示，而非原始输入的不同部分
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码(RoPE)到查询和键向量
        # RoPE通过复数旋转为每个token编码其位置信息
        # 位置编码应用于每个头处理的投影向量，而非原始输入的不同部分
        # 每个头接收相同的位置编码公式，但应用于不同的投影子空间
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        
        # 处理KV缓存
        if past_key_value is not None:
            # 将当前键值与缓存连接
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 更新KV缓存
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序，并重复键值以匹配查询头数
        xq, xk, xv = (
            xq.transpose(1, 2),  # [batch_size, n_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 重复键
            repeat_kv(xv, self.n_rep).transpose(1, 2)   # 重复值
        )
        
        # 使用Flash Attention或传统注意力计算
        if self.flash and seq_len != 1:
            # Flash Attention实现（PyTorch 2.0+）
            # 训练时使用设定的dropout率，推理时不使用dropout
            dropout_p = self.dropout if self.training else 0.0
            
            # 使用PyTorch内置的scaled_dot_product_attention函数实现Flash Attention
            # - xq: 查询张量 [batch_size, n_heads, seq_len, head_dim]
            # - xk: 键张量 [batch_size, n_heads, seq_len, head_dim]
            # - xv: 值张量 [batch_size, n_heads, seq_len, head_dim]
            # - attn_mask=None: 不需要显式提供注意力掩码，因为使用is_causal=True
            # - dropout_p: 应用于注意力权重的dropout概率
            # - is_causal=True: 启用因果掩码，确保每个token只能关注自身及之前的token
            #
            # Flash Attention是一种内存高效的注意力算法，通过分块计算和重用中间结果
            # 来减少内存占用和提高计算速度。它特别适合长序列和大批量的情况。
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,  # 不需要显式提供注意力掩码，因为is_causal=True参数会自动创建上三角掩码
                # Flash Attention通过is_causal=True参数实现因果掩码，确保每个token只能关注自身及之前的token
                # 这种实现方式比传统掩码更高效，因为它在算法内部优化了掩码的应用，无需额外的内存开销
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # 传统注意力实现
            # 计算注意力分数
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 应用因果掩码
            scores += self.mask[:, :, :seq_len, :seq_len]
            # Softmax归一化
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 应用dropout
            scores = self.attn_dropout(scores)
            # 计算加权和
            output = scores @ xv

        # 重塑并投影回原始维度
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络，使用SwiGLU激活函数
    """
    def __init__(self, config: LMConfig):
        """
        初始化前馈网络
        
        参数:
            config: 模型配置
        """
        super().__init__()
        # 如果未指定隐藏维度，则计算默认值
        if config.hidden_dim is None:
            # 1. 首先将隐藏层维度设为输入维度的4倍（标准FFN通常是4倍）
            hidden_dim = 4 * config.dim
            # 2. 然后将其缩小到原来的2/3（SwiGLU激活函数的优化设置）
            hidden_dim = int(2 * hidden_dim / 3)
            # 3. 最后将隐藏维度调整为multiple_of的整数倍，以提高硬件效率
            # 这里使用向上取整确保维度不会小于计算值
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # 定义三个线性层
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 上投影
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # 下投影
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 门控投影
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        # SwiGLU激活: (SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    """
    混合专家门控机制，用于选择激活哪些专家
    """
    def __init__(self, config: LMConfig):
        """
        初始化门控机制
        
        参数:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数
        self.n_routed_experts = config.n_routed_experts  # 可路由专家数量
        
        self.scoring_func = config.scoring_func  # 评分函数类型
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux  # 是否使用序列级辅助损失
        
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化top-k概率
        self.gating_dim = config.dim  # 门控维度
        # 门控权重矩阵
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化门控权重参数"""
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播
        
        参数:
            hidden_states: 输入隐藏状态
            
        返回:
            专家索引、专家权重和辅助损失
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个专家的路由分数
        logits = F.linear(hidden_states, self.weight, None)
        
        # 应用评分函数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'不支持的MoE门控评分函数: {self.scoring_func}')

        # 选择top-k专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化top-k权重
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（负载均衡）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            # 序列级辅助损失
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 计算每个专家的使用频率
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算辅助损失
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 标准辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)  # 每个专家的平均使用率
                Pi = scores_for_aux.mean(0)   # 每个专家的平均路由概率
                fi = ce * self.n_routed_experts  # 专家使用频率
                # 计算辅助损失
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
            
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    混合专家前馈网络，实现稀疏MoE架构
    """
    def __init__(self, config: LMConfig):
        """
        初始化MoE层
        
        参数:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控机制
        self.gate = MoEGate(config)
        # 可选的共享专家（始终激活）
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练模式：为每个选中的专家复制输入
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 对每个专家分别处理相应的输入
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
                
            # 加权组合专家输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用优化的推理方法
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
            
        # 添加共享专家的输出（如果有）
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
            
        # 保存辅助损失供外部访问
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化的MoE推理实现
        
        参数:
            x: 输入张量
            flat_expert_indices: 展平的专家索引
            flat_expert_weights: 展平的专家权重
            
        返回:
            专家输出的加权和
        """
        expert_cache = torch.zeros_like(x)
        # 按专家索引排序
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的token数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 获取token索引
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 对每个专家分别处理
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
                
            expert = self.experts[i]
            # 获取当前专家处理的token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 提取相应的输入
            expert_tokens = x[exp_token_idx]
            # 专家处理
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 应用权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到输出缓存
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMind模型的基本构建块，包含注意力层和前馈网络
    """
    def __init__(self, layer_id: int, config: LMConfig):
        """
        初始化模型块
        
        参数:
            layer_id: 层索引
            config: 模型配置
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        # 多头注意力层
        self.attention = Attention(config)

        self.layer_id = layer_id
        # 层归一化
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 前馈网络（普通FFN或MoE）
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        """
        前向传播
        
        参数:
            x: 输入张量
            pos_cis: 位置编码
            past_key_value: KV缓存
            use_cache: 是否使用KV缓存
            
        返回:
            输出张量和更新的KV缓存
        """
        # 注意力层（带残差连接）
        h_attn, past_kv = self.attention(
            self.attention_norm(x),  # 预层归一化
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        # 残差连接：将原始输入x直接加到注意力层输出h_attn上
        # 作用1：缓解梯度消失问题，使深层网络更容易训练
        # 作用2：提供信息捷径，让模型可以保留原始特征
        # 作用3：使模型更加平滑，提高泛化能力和稳定性
        h = x + h_attn  # 残差连接
        
        # 前馈网络（带残差连接）
        # 同样的残差连接应用于前馈网络：将前一层输出h直接加到FFN输出上
        # 这种"预层归一化+残差连接"的结构是Transformer架构的关键设计
        # 每个子层(注意力/FFN)都有自己的残差路径，确保信息可以有效地流过深层网络
        out = h + self.feed_forward(self.ffn_norm(h))  # 预层归一化
        
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    """
    MiniMind语言模型主类，继承自HuggingFace的PreTrainedModel
    """
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        """
        初始化模型
        
        参数:
            params: 模型配置
        """
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        
        # Token嵌入层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        # 创建多层Transformer块
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        
        # 最终层归一化
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        # 输出投影（与嵌入层权重共享）
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        
        # 预计算旋转位置编码(RoPE)
        # - dim=params.dim // params.n_heads: 每个注意力头的维度(head_dim)
        #   注意：这里传入head_dim而非dim，因为位置编码应用于多头注意力中每个头处理的投影向量
        # - theta=params.rope_theta: 频率衰减参数，通常为10000
        # 预计算可以提高效率，避免在每次前向传播时重新计算
        # 注册为非持久缓冲区(persistent=False)，不会被保存到模型检查点中
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
                             
        # 输出容器
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = True,
                **args):
        """
        模型前向传播
        
        参数:
            input_ids: 输入token ID
            past_key_values: KV缓存
            use_cache: 是否使用KV缓存
            args: 其他参数
            
        返回:
            包含logits、辅助损失和KV缓存的输出对象
        """
        # 初始化KV缓存
        past_key_values = past_key_values or [None] * len(self.layers)
        # 获取起始位置（用于KV缓存和位置编码）
        # 在生成过程中，start_pos表示已生成序列的长度，用于正确应用位置编码
        start_pos = args.get('start_pos', 0)
        
        # Token嵌入
        h = self.dropout(self.tok_embeddings(input_ids))
        
        # 获取当前序列的位置编码
        # 从预计算的位置编码中切片出当前序列需要的部分
        # 位置编码的维度是head_dim而非dim，因为它将应用于多头注意力中每个头处理的投影向量
        # 每个头接收相同的位置编码，但应用于不同的投影子空间
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        
        # 存储KV缓存
        past_kvs = []
        
        # 通过每一层
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
            
        # 最终层归一化和输出投影
        logits = self.output(self.norm(h))
        
        # 计算MoE辅助损失（如果使用MoE）
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        
        # 设置输出对象的属性
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        """
        生成文本
        
        参数:
            input_ids: 输入token ID
            eos_token_id: 结束符ID
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            stream: 是否流式生成
            rp: 重复惩罚系数
            use_cache: 是否使用KV缓存
            pad_token_id: 填充符ID
            args: 其他参数
            
        返回:
            生成的token序列
        """
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 批量生成
        generated = []
        for i in range(input_ids.size(0)):
            # 移除填充符
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # 流式生成
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 收集生成的token
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            # 拼接完整序列
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
            
        # 填充到相同长度
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
