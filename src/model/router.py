import torch
from torch import nn
import torch.nn.functional as F
"""
Mixture of Experts (MoE) Gating Networks with PyTorch

This module implements several gating mechanisms for MoE models:
1. Basic softmax gating (GatingNetwork)
2. Top-k gating with Gumbel softmax (TopKGatingNetwork)
3. Noisy top-k gating with learned noise (NoisyTopKGatingNetwork)

Usage:
1. Basic gating:
   net = GatingNetwork(num_skill_experts=4, hidden_size=1024)
   gates = net(trajectory_embeds)  # [batch, seq_len, num_experts]

2. Top-k gating:
   net = TopKGatingNetwork(top_k=2, num_skill_experts=6, hidden_size=1024, seq_len=10)
   gates = net(trajectory_embeds)  # [batch, num_experts]

3. Noisy top-k gating:
   net = NoisyTopKGatingNetwork(top_k=2, num_skill_experts=6, hidden_size=1024, seq_len=10)
   gates = net(trajectory_embeds)  # [batch, num_experts]
"""


def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def soft_topk(logits, k, temperature=1.0):
    noise = sample_gumbel(logits.shape, device=logits.device)
    perturbed_logits = logits + noise
    soft_probs = F.softmax(perturbed_logits / temperature, dim=-1)
    topk_vals, _ = torch.topk(perturbed_logits, k, dim=-1)
    threshold = topk_vals[..., -1].unsqueeze(-1)
    mask = torch.sigmoid((perturbed_logits - threshold) / temperature)
    soft_topk_probs = soft_probs * mask
    soft_topk_probs = soft_topk_probs / soft_topk_probs.sum(dim=-1, keepdim=True)
    return soft_topk_probs


class GatingNetwork(nn.Module):
    def __init__(self, num_skill_experts, hidden_size):
        super().__init__()
        self.num_experts = num_skill_experts
        self.gate = nn.Linear(hidden_size, self.num_experts)

    def forward(self, x):
        gate_values = self.gate(x)
        gate_values = torch.softmax(gate_values, dim=-1)
        return gate_values

class TopKGatingNetwork(nn.Module):
    def __init__(self, top_k, num_skill_experts, hidden_size, seq_len):
        super().__init__()
        self.num_experts = num_skill_experts
        self.gate = nn.Linear(hidden_size * seq_len, self.num_experts)
        self.top_k = top_k
    
    def forward(self, x): # [batch_size, seq_len, hidden_size] -> [batch_size, num_of_experts]
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(batch_size, -1) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len * hidden_size]
        logits = self.gate(x) # [batch_size, seq_len * hidden_size] -> [batch_size, num_of_experts]
        
        gate_values = soft_topk(logits, self.top_k)
        # top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        # zeros = torch.full_like(logits, float('-inf'))
        # sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # gate_values = F.softmax(sparse_logits, dim=-1)
        return gate_values

class NoisyTopKGatingNetwork(nn.Module):
    def __init__(self, top_k, num_skill_experts, hidden_size, seq_len):
        super().__init__()
        self.num_experts = num_skill_experts
        self.gate = nn.Linear(hidden_size * seq_len, self.num_experts)
        self.noise_linear =nn.Linear(hidden_size * seq_len, self.num_experts)
        self.top_k = top_k
    
    def forward(self, x): # [batch_size, seq_len, hidden_size] -> [batch_size, num_of_experts]
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(batch_size, -1) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len * hidden_size]
        logits = self.gate(x) # [batch_size, seq_len * hidden_size] -> [batch_size, num_of_experts]
        noise_logits = self.noise_linear(x) # [batch_size, seq_len * hidden_size] -> [batch_size, num_of_experts]
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        
        gate_values = soft_topk(noisy_logits, self.top_k)
        # top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # zeros = torch.full_like(noisy_logits, float('-inf'))
        # sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # gate_values = F.softmax(sparse_logits, dim=-1)
        return gate_values


if __name__ == '__main__':
    import torch
    from omegaconf import OmegaConf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    # Model parameters
    num_experts = 4
    top_k = 2
    batch_size = 3
    seq_len = 10
    hidden_size = 1024

    # Generate dummy data
    trajectory_embeds = torch.randn(batch_size, seq_len, hidden_size).to(device)  # [batch_size, seq_len, hidden_size]

    net_1 = GatingNetwork(num_skill_experts=num_experts, hidden_size=hidden_size).to(device)
    net_2 = TopKGatingNetwork(
        top_k=top_k,
        num_skill_experts=6,
        hidden_size=hidden_size,
        seq_len=seq_len
    ).to(device)


    with torch.no_grad():
        Y_1 = net_1(trajectory_embeds)  # [batch_size, num_experts]
        Y_2 = net_2(trajectory_embeds)  # [batch_size, num_experts]

    print(f"GatingNetwork output shape: {Y_1.shape}")
    print(f"TopKGatingNetwork output shape: {Y_2.shape}")

    assert Y_1.shape == (batch_size, seq_len, num_experts), "GatingNetwork output shape mismatch"
    assert Y_2.shape == (batch_size, 6), "TopKGatingNetwork output shape mismatch"
    
##################################################################################################
# Using device: cuda
# GPU detected: NVIDIA GeForce RTX 4090
# CUDA version: 12.4
# Available VRAM: 51.00 GB
# GatingNetwork output shape: torch.Size([3, 10, 4])
# TopKGatingNetwork output shape: torch.Size([3, 6])