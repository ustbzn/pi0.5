"""
Shared Expert + Routed Expert MoE MLP (DeepSeek-V2 风格)

设计目标
--------
1. 每个 token 输出 = shared_expert(x) + gate_weight * routed_expert_k(x)
   - shared_expert：始终参与计算，承载预训练 / 已训练的稳定信息流，类似 residual。
   - routed_expert：从 N 个里选 top-1，学习专门化特征。
   - 训练初期 routed_expert 的 ``down_proj`` 被零初始化（见
     ``init_routed_from_shared``），routed 分支严格输出 0，保证第 0 步等价 baseline。
2. Top-1 路由的梯度路径：``forward`` 中只在 ``top_k > 1`` 时归一化 top-k 概率；
   ``top_k == 1`` 时保留未归一化的 softmax 概率作为乘子，让主 loss 通过
   ``expert_output * gate_prob`` 把任务梯度回传到 router gate。
3. 与 v2 原始设计文档相比，本实现做了 3 处工程修正以适配当前代码：
   a. forward 仍然返回 tensor（不返回 tuple），aux_loss 写入模块属性 ``_pending_aux_loss``，
      由 ``DrivePiZero.forward`` 末尾统一收集——这样 ``layer_func("mlp", ...)`` 接口保持
      不变，``torch.compile`` 不需要处理多返回值边界。
   b. mlp_class 默认应用 Qwen3MLP（当前代码 action mixture 用的就是 Qwen3MLP，而不是
      早期文档中的 GemmaMLP）。
   c. 内部对未被路由到的 routed expert 做一次乘 0 的 dummy forward，保证它们的参数仍
      有非 None grad，避免 DDP（未开启 ``find_unused_parameters``）报错。

依赖
----
仅依赖 PyTorch 与外部传入的 mlp_class、config，不引用具体 backbone，便于单测。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedExpertMoEMLP(nn.Module):
    """1 shared + N routed top-k MoE MLP。

    Args:
        mlp_class:           构造 expert 用的 MLP 类（如 Qwen3MLP / GemmaMLP）。
        config:              传给 mlp_class(config, **mlp_kwargs) 的配置对象，
                             需要包含 ``hidden_size`` 属性。
        num_routed_experts:  routed expert 数量，默认 3。
        top_k:               每个 token 从 routed experts 中选几个，默认 1。
        load_balance_coeff:  Switch Transformer 风格的负载均衡 loss 系数，
                             仅作用于 routed experts。
        mlp_kwargs:          透传给 mlp_class 的额外参数（如 ``use_quantize`` / ``use_lora``）。
    """

    def __init__(
        self,
        mlp_class,
        config,
        num_routed_experts: int = 3,
        top_k: int = 1,
        load_balance_coeff: float = 0.01,
        mlp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if num_routed_experts < 1:
            raise ValueError(f"num_routed_experts must be >= 1, got {num_routed_experts}")
        if top_k < 1 or top_k > num_routed_experts:
            raise ValueError(
                f"top_k={top_k} must be in [1, num_routed_experts={num_routed_experts}]"
            )

        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff

        mlp_kwargs = dict(mlp_kwargs or {})

        # 1) Shared Expert：始终参与，承担 residual 角色。
        self.shared_expert = mlp_class(config, **mlp_kwargs)

        # 2) Routed Experts：通过 router 选 top-k。
        self.routed_experts = nn.ModuleList([
            mlp_class(config, **mlp_kwargs) for _ in range(num_routed_experts)
        ])

        # 3) Router：将 hidden_states 映射到 routed expert 概率，
        #    零初始化保证训练初期均匀路由。
        self.gate = nn.Linear(config.hidden_size, num_routed_experts, bias=False)
        nn.init.zeros_(self.gate.weight)

        # forward 时累加用：
        #   - _pending_aux_loss：保留 grad 的 tensor，每次 forward 重写，
        #     由 DrivePiZero.forward 末尾收集后清空。
        #   - _last_aux_loss_value：用于 log 的 float 值（无 grad）。
        self._pending_aux_loss: Optional[torch.Tensor] = None
        self.register_buffer(
            "_last_aux_loss_value",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        # 监控：最近一次 forward 中每个 routed expert 收到的 token 比例
        self.register_buffer(
            "_last_expert_usage",
            torch.zeros(num_routed_experts, dtype=torch.float32),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # 初始化辅助：用于从已训练 MLP 权重启动 shared / routed expert。
    # ------------------------------------------------------------------
    def init_shared_from_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """用给定的 state_dict 初始化 shared_expert。

        典型用法：在 ``DrivePiZero.__init__`` 完成 MLP → SharedExpertMoEMLP 的替换之前，
        先把原 MLP 的 state_dict 缓存下来，替换后调用本方法。
        """
        self.shared_expert.load_state_dict(state_dict, strict=True)

    def init_routed_from_shared(self, noise_std: float = 1e-3):
        """用 shared_expert 的权重初始化每个 routed expert，并把 routed 分支的
        最后一层 (down_proj) 置零，使 routed_expert(x) ≡ 0 在训练第 0 步严格成立。

        语义: 训练初期 ``out = shared(x) + routed(x) = shared(x)``，完整保留
        baseline；routed 分支等价于一个零初始化的 delta head，由训练自然学出
        specialization，不会引发跨 14 层的残差流漂移。

        - ``gate_proj`` / ``up_proj`` 仍从 shared 拷贝并叠加小幅高斯噪声，用于
          打破 N 个 routed expert 之间的对称性（否则一旦 down_proj 开始学，
          多个 expert 会沿相同方向同步演化）。Qwen3MLP 这类线性层权重 std
          ≈ 1/√fan_in ≈ 0.02–0.03，因此 ``noise_std=1e-3`` 已经是几个百分比
          的相对扰动，足够破对称。
        - ``down_proj.weight`` 强制置零；Qwen3MLP 中 ``bias=False`` 不必再处理偏置。
        - 若需要严格镜像 shared（不打破对称性，仅作 sanity check），传 ``noise_std=0.0``。
        """
        shared_state = self.shared_expert.state_dict()
        for expert in self.routed_experts:
            expert.load_state_dict(shared_state, strict=True)
            with torch.no_grad():
                # 仅在 gate_proj / up_proj 上加噪声，down_proj 待会儿会被清零
                if noise_std > 0:
                    for name, p in expert.named_parameters():
                        if name.startswith("down_proj"):
                            continue
                        p.add_(torch.randn_like(p) * noise_std)
                # routed 分支最后一层置零 → routed(x) = 0 at step 0
                # forward: down_proj(silu(gate_proj(x)) * up_proj(x))
                # 当 down_proj.weight = 0 时输出严格为零，与 gate_proj/up_proj 上
                # 的扰动无关，因此 baseline 完整保留。
                for name, p in expert.named_parameters():
                    if name.startswith("down_proj"):
                        p.zero_()

    # ------------------------------------------------------------------
    # 前向：返回 tensor，aux_loss 走 _pending_aux_loss 由外部收集。
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        num_tokens = x_flat.shape[0]

        # 1) Shared Expert：始终计算
        shared_out = self.shared_expert(x_flat)

        # 2) Router 概率
        # gate 用 float32 计算更稳定，再 cast 回原 dtype 用于加权
        logits = self.gate(x_flat.to(self.gate.weight.dtype))
        probs = F.softmax(logits, dim=-1)  # [N, num_routed_experts]

        # 3) Top-k selection
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # 仅 k>1 时做归一化让加权和=1；k=1 时若也归一化则 weight 恒等于 1，
        # 主 loss 的梯度无法回传到 router（gate 只剩 load-balance loss 推动，
        # 容易学成均匀分流）。k=1 时直接用未归一化的 softmax 概率作为乘子，
        # 让 chosen expert 的输出沿 gate_prob 把梯度送回 router。
        if self.top_k > 1:
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 4) Routed expert 计算
        # 为了节省显存采用稀疏 dispatch；同时保证每个 routed expert 都有非 None grad
        # （便于 DDP 不开 find_unused_parameters 也能跑）。
        routed_out = torch.zeros_like(x_flat)
        selected_experts = set()
        for e in range(self.num_routed_experts):
            mask = (topk_indices == e)            # [N, top_k]
            token_mask = mask.any(dim=-1)         # [N]
            if not token_mask.any():
                continue
            selected_experts.add(e)
            expert_input = x_flat[token_mask]
            expert_output = self.routed_experts[e](expert_input)
            # 每个 token 对该 expert 的总权重（top-k 求和，k=1 时即唯一概率）
            weight = (topk_probs * mask.to(topk_probs.dtype)).sum(dim=-1)  # [N]
            routed_out[token_mask] = (
                routed_out[token_mask]
                + weight[token_mask].to(expert_output.dtype).unsqueeze(-1) * expert_output
            )

        # 4b) Dummy forward：未被路由到的 routed expert 仍要有梯度路径，
        #     做一次 batch=1 的虚拟 forward，乘 0 后加进 routed_out，
        #     数值上完全不影响输出，但保证参数 grad 不为 None。
        for e in range(self.num_routed_experts):
            if e in selected_experts:
                continue
            dummy_input = x_flat[:1].detach()  # detach 避免向 x 回传冗余 grad
            dummy_output = self.routed_experts[e](dummy_input)
            routed_out[:1] = routed_out[:1] + dummy_output * 0.0

        out = shared_out + routed_out

        # 5) 负载均衡 aux loss（仅对 routed experts）
        aux_loss = self._load_balance_loss(probs, topk_indices, num_tokens=num_tokens)
        self._pending_aux_loss = aux_loss
        self._last_aux_loss_value = aux_loss.detach().to(self._last_aux_loss_value.dtype)

        # 6) 监控：每个 expert 实际收到的 token 比例（无梯度）
        with torch.no_grad():
            usage = torch.zeros(
                self.num_routed_experts, device=x.device, dtype=torch.float32
            )
            for k in range(self.top_k):
                usage = usage + torch.bincount(
                    topk_indices[:, k], minlength=self.num_routed_experts
                ).to(usage.dtype)
            self._last_expert_usage = usage / max(num_tokens * self.top_k, 1)

        return out.reshape(B, S, D)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _load_balance_loss(
        self,
        probs: torch.Tensor,        # [N, num_routed_experts]
        topk_indices: torch.Tensor, # [N, top_k]
        num_tokens: int,
    ) -> torch.Tensor:
        """Switch Transformer 风格负载均衡 loss：

            L_aux = coeff * N_e * sum_i (f_i * P_i)

        - f_i：分配给 routed expert i 的 token 占比（detach，不回传）
        - P_i：所有 token 对 routed expert i 的平均 routing 概率（带梯度）
        """
        # f_i：不需要回传梯度（counts 是离散统计）
        counts = torch.zeros(
            self.num_routed_experts, device=probs.device, dtype=probs.dtype
        )
        for k in range(self.top_k):
            counts = counts + torch.bincount(
                topk_indices[:, k], minlength=self.num_routed_experts
            ).to(probs.dtype)
        f = counts / max(num_tokens * self.top_k, 1)

        # P_i：保留梯度
        P = probs.mean(dim=0)

        return self.load_balance_coeff * self.num_routed_experts * (f.detach() * P).sum()

    # ------------------------------------------------------------------
    # 辅助：清空 aux_loss 缓存（forward 后由 DrivePiZero 统一调用）
    # ------------------------------------------------------------------
    def consume_aux_loss(self) -> Optional[torch.Tensor]:
        """取出 ``_pending_aux_loss`` 并清空。

        返回 None 表示当次 forward 没有产生 aux_loss（理论上不会发生，除非
        forward 没被调用过；保留 None 分支做防御）。
        """
        aux = self._pending_aux_loss
        self._pending_aux_loss = None
        return aux

    def last_aux_loss_value(self) -> float:
        """供日志读取的 float 值。"""
        return float(self._last_aux_loss_value.item())

    def last_expert_usage(self) -> torch.Tensor:
        """返回最近一次 forward 中每个 routed expert 的 token 比例，形状 [num_routed_experts]。"""
        return self._last_expert_usage.detach().clone()
