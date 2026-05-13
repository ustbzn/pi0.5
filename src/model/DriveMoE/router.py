import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, token_dim=768, num_labels=6, num_heads=8):
        super().__init__()
        self.token_dim = token_dim
        
        self.waypoint_projector = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, token_dim)
        )
        
        self.router_query = nn.Parameter(torch.randn(1, 1, token_dim))
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.ln = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim // 2, num_labels)
        )

    def forward(self, front_tokens, waypoint):
        """
        Args:
            front_tokens: [Batch, Num_Tokens, Dim]
            waypoint: [Batch, 2]
        Returns:
            logits: [Batch, Num_Cameras]
        """
        batch_size = front_tokens.size(0)
        
        # wp_token shape: [Batch, 1, Dim]
        wp_token = self.waypoint_projector(waypoint).unsqueeze(1)
        
        # combined_kv shape: [Batch, Num_Tokens + 1, Dim]
        combined_kv = torch.cat([wp_token, front_tokens], dim=1)
        
        query = self.router_query.expand(batch_size, -1, -1)
        
        attn_out, _ = self.cross_attn(
            query=query, 
            key=combined_kv, 
            value=combined_kv
        )
        
        x = self.ln(attn_out.squeeze(1)) # [Batch, Dim]
        logits = self.mlp(x)
        return logits

if __name__ == "__main__":
    model = Router(token_dim=768, num_labels=6)
    front_vit_features = torch.randn(8, 256, 768)
    waypoints = torch.randn(8, 2)

    logits = model(front_vit_features, waypoints)
    print(f"Logits shape: {logits.shape}")