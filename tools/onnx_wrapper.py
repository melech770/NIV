"""Traceable ONNX wrapper with dynamic sequence length.

torch.onnx's legacy tracer bakes ``Tensor.size()`` results into reshape nodes, which
breaks variable-length glyphs.  Everything here is written with shape-agnostic ops
(``unflatten``, ``permute``, ``flatten``, explicit positional encodings) so the
exported graph works for any number of points.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class NivOnnxModule(torch.nn.Module):
    """points[B,N,4], pad_mask[B,N], positions[B,N], combo_idx[K], combo_w[K] -> [B,N,2]."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.point_mlp = model.point_mlp
        self.layers = model.cond_transformer_encoder.layers
        self.delta_head = model.delta_head
        self.embed = model.axis_id_and_value_embedder.axis_id_to_embedder
        self.num_combos = float(model.axis_id_and_value_embedder.num_combos)
        d_model = int(model.point_mlp[2].out_features)
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def _positional(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.unsqueeze(-1) * self.div_term.unsqueeze(0).unsqueeze(0)
        interleaved = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        return interleaved.flatten(-2)

    def _attention(self, layer: torch.nn.Module, hidden: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        weight = layer.self_attn.in_proj_weight
        bias = layer.self_attn.in_proj_bias
        heads = layer.self_attn.num_heads
        head_dim = self.d_model // heads
        qkv = F.linear(hidden, weight, bias)
        query, key, value = qkv.unflatten(-1, (3, heads, head_dim)).permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), -1e4)
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        context = context.permute(0, 2, 1, 3).flatten(2, 3)
        out_proj = layer.self_attn.out_proj
        return F.linear(context, out_proj.weight, out_proj.bias)

    def forward(
        self,
        points: torch.Tensor,
        pad_mask: torch.Tensor,
        positions: torch.Tensor,
        combo_idx: torch.Tensor,
        combo_w: torch.Tensor,
    ) -> torch.Tensor:
        cond = (self.embed.weight[combo_idx] * combo_w.unsqueeze(-1)).sum(0, keepdim=True) / self.num_combos
        hidden = self.point_mlp(points) + self._positional(positions)

        for layer in self.layers:
            hidden = hidden + self._attention(layer, hidden, pad_mask)
            hidden = layer.norm1(hidden, cond)
            hidden = hidden + layer.linear2(layer.activation(layer.linear1(hidden)))
            hidden = layer.norm2(hidden, cond)

        return self.delta_head(hidden)
