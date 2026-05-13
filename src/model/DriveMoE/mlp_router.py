import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRouter(nn.Module):
    def __init__(self, hidden_size=1024, num_experts=7, use_noisy_top_k=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.use_noisy_top_k = use_noisy_top_k
        
        self.router_linear = nn.Linear(hidden_size, num_experts, bias=False)
        
        if use_noisy_top_k:
            self.noise_linear = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.router_linear.weight, std=0.02)
        if self.use_noisy_top_k:
            nn.init.trunc_normal_(self.noise_linear.weight, std=0.02)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len=10, hidden_size=1024]
        
        Returns:
            router_logits: [batch_size, num_experts]
        """
        _, seq_len, hidden_size = hidden_states.shape
        assert hidden_size == self.hidden_size, f"Expected hidden_size {self.hidden_size}, got {hidden_size}"
        
        token_logits = self.router_linear(hidden_states)  # [bs, 10, num_experts]
        
        if self.use_noisy_top_k and self.training:
            noise = self.noise_linear(hidden_states)  # [bs, 10, num_experts]
            noise = torch.randn_like(noise) * F.softplus(noise)
            token_logits = token_logits + noise
        
        router_logits = token_logits.mean(dim=1)  # [bs, num_experts]
        
        router_logits = router_logits / self.temperature.abs()
        
        return router_logits