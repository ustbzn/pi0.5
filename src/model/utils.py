import torch


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """
    x:   [B, H, S, D]
    cos: [B, S, D] → after unsqueeze(1) → [B, 1, S, D]
    """
    # ── 防御：若 cos 仍是 3D（旧调用路径），兼容处理 ─────────────────
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # ── Batch 不匹配时的最终保险 ────────────────────────────────────
    if cos.shape[0] != x.shape[0]:
        raise RuntimeError(
            f"apply_rotary_pos_emb: batch mismatch! "
            f"x={x.shape}, cos={cos.shape}. "
            f"Root cause: forward_rotary_emb 返回的 cos/sin batch 维与 hidden_states 不匹配，"
            f"请检查 Mixture.forward_rotary_emb 的修复是否生效。"
        )
    # cos = cos.to(dtype=x.dtype)
    # sin = sin.to(dtype=x.dtype)
    
    # cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, S, D]
    # sin = sin.unsqueeze(unsqueeze_dim)
    # print(f"apply_rotary_pos_emb: x shape {x.shape}, cos shape {cos.shape}, sin shape {sin.shape}")
    # print((x * cos).shape, (rotate_half(x) * sin).shape)
    # import pdb; pdb.set_trace()
    x = (x * cos) + (rotate_half(x) * sin)
    return x

# def rotate_half(x):
#     x = x.view(*x.shape[:-1], -1, 2)
#     x1, x2 = x.unbind(dim=-1)
#     return torch.stack((-x2, x1), dim=-1).flatten(-2)

# def apply_rotary_pos_emb(states, cos, sin):
#     """
#     states: [B, H, S, D]
#     cos,sin: [B, S, D]
#     """

#     cos = cos.unsqueeze(1)  # [B,1,S,D]
#     sin = sin.unsqueeze(1)

#     states = (states * cos) + (rotate_half(states) * sin)

#     return states
# def apply_rotary_pos_emb(q, k, cos, sin):
#     # q,k: [B,H,S,D]
#     # cos,sin: [B,S,D]

#     cos = cos.unsqueeze(1)  # [B,1,S,D]
#     sin = sin.unsqueeze(1)

#     q = (q * cos) + (rotate_half(q) * sin)
#     k = (k * cos) + (rotate_half(k) * sin)

#     return q, k


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )