#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Transformer Layers
#
#################################################################################################################

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops

from typing import Any

from src.nilmformer.congif import NILMFormerConfig


class DiagonalMaskFromSeqlen:
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            self._mask = torch.diag(
                torch.ones(L, dtype=torch.bool, device=device)
            ).repeat(B, 1, 1, 1)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask # [B, 1, L, L]


class DiagonnalyMaskedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
        use_efficient_attention: bool = False,
    ):
        super().__init__()

        self.use_efficient_attention: bool = use_efficient_attention

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.dropout: float = dropout

        self.scale = head_dim**-0.5

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch, seqlen, self.n_heads, self.head_dim) # [B, L, N_H, H]
        xk = xk.view(batch, seqlen, self.n_heads, self.head_dim) # [B, L, N_H, H]
        xv = xv.view(batch, seqlen, self.n_heads, self.head_dim) # [B, L, N_H, H]

        diag_mask = DiagonalMaskFromSeqlen(batch, seqlen, device=xq.device) # [B, 1, L, L]

        if self.use_efficient_attention:
            output = xops.memory_efficient_attention(
                xq,
                xk,
                xv,
                attn_bias=(diag_mask.mask).repeat(1, self.n_heads, 1, 1).float(),
                p=self.dropout,
            )
        else:
            scale = 1.0 / xq.shape[-1] ** 0.5
            scores = torch.einsum("blhe,bshe->bhls", xq, xk)
            attn = self.attn_dropout(
                torch.softmax(
                    scale * scores.masked_fill_(diag_mask.mask, torch.finfo(scores.dtype).min), dim=-1
                )
            )
            output = torch.einsum("bhls,bshd->blhd", attn, xv)

        return self.out_dropout(self.wo(output.reshape(batch, seqlen, -1)))


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dp_rate: float = 0.0,
        activation: Any = F.gelu,
        bias1: bool = True,
        bias2: bool = True,
    ):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = activation

    def forward(self, x) -> torch.Tensor:
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, NFconfig: NILMFormerConfig):
        super().__init__()
        assert not NFconfig.d_model % NFconfig.n_head, (
            f"d_model ({NFconfig.d_model}) must be divisible by n_heads ({NFconfig.n_head})"
        )

        self.attention_layer = DiagonnalyMaskedSelfAttention(
            dim=NFconfig.d_model,
            n_heads=NFconfig.n_head,
            head_dim=NFconfig.d_model // NFconfig.n_head,
            dropout=NFconfig.dp_rate,
            use_efficient_attention=NFconfig.use_efficient_attention,
        )

        self.norm1 = nn.LayerNorm(NFconfig.d_model, eps=NFconfig.norm_eps)
        self.norm2 = nn.LayerNorm(NFconfig.d_model, eps=NFconfig.norm_eps)

        self.dropout = nn.Dropout(NFconfig.dp_rate)

        self.pffn = PositionWiseFeedForward(
            dim=NFconfig.d_model,
            hidden_dim=NFconfig.d_model * NFconfig.pffn_ratio,
            dp_rate=NFconfig.dp_rate,
        )

    def forward(self, x) -> torch.Tensor:
        # x input and output shape [batch, seq_length, d_model] to meet Transformer Convention

        # Attention Block
        x = self.norm1(x)
        new_x = self.attention_layer(x)
        x = torch.add(x, new_x)

        # PFFN Block
        x = self.norm2(x)
        new_x = self.pffn(x)
        x = torch.add(x, self.dropout(new_x))

        return x
