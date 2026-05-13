import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha.detach().clone())
            else:
                self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            at = self.alpha.gather(0, targets.data)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class CombinedLoss(nn.Module):
    def __init__(self, camera_router_weight, action_router_weight, action_weight, gamma):
        super().__init__()
        self.camera_router_weight = camera_router_weight
        self.action_router_weight = action_router_weight
        self.action_weight = action_weight
        camera_counts = torch.tensor([35105, 8865, 161717, 13671, 5990], dtype=torch.float)
        camera_alpha = (1.0 / camera_counts)
        camera_alpha = camera_alpha / camera_alpha.sum()
        action_counts = torch.tensor([13304, 1036, 29921, 12064, 5200, 45332, 118491], dtype=torch.float)
        action_alpha = (1.0 / action_counts)
        action_alpha = action_alpha / action_alpha.sum()
        self.camera_router_criterion = FocalLoss(alpha=camera_alpha, gamma=gamma)
        self.action_router_criterion = FocalLoss(alpha=action_alpha, gamma=gamma)

    forward_call = nn.Module.forward
    
    def _compute_camera_loss(
        self,
        camera_logits: torch.Tensor,
        camera_ids: torch.Tensor,
    ):
        target_ids = camera_ids.long()
        return self.camera_router_criterion(camera_logits, target_ids)
    
    def _compute_action_router_loss(
        self,
        action_logits: torch.Tensor,
        scenario_ids: torch.Tensor,
    ):
        target_ids = scenario_ids.long()
        return self.action_router_criterion(action_logits, target_ids)
    
    def _compute_action_flowmatching_loss(
        self,
        flow_sig_min: float,
        x0: torch.Tensor,
        x1: torch.Tensor,
        v_psi: torch.Tensor,
    ):
        # compare to true velocity
        d_psi = x1 - (1 - flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)
        

    def forward(
        self, 
        flow_sig_min: float,
        x0: torch.Tensor,
        x1: torch.Tensor,
        v_psi: torch.Tensor,
        camera_logits: torch.Tensor,
        action_logits: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
        scenario_ids: Optional[torch.Tensor] = None,
    ):
        camera_loss = None
        action_router_loss = None
        
        action_loss = self._compute_action_flowmatching_loss(flow_sig_min, x0, x1, v_psi)
        total_loss = self.action_weight * action_loss
        
        if camera_ids is not None:
            camera_loss = self._compute_camera_loss(camera_logits, camera_ids) * self.camera_router_weight
            total_loss += camera_loss
        if scenario_ids is not None:
            action_router_loss = self._compute_action_router_loss(action_logits, scenario_ids) * self.action_router_weight
            total_loss += action_router_loss
        
        return total_loss, action_loss, camera_loss, action_router_loss