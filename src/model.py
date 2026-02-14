import torch
import torch.nn as nn

# ==========================================
# Expert Network
# ==========================================
class Expert(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(Expert, self).__init__()
        # Simple linear expert as described in the paper
        # Can only approximate linear relationships (cannot fit complex curves alone)
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)

# ==========================================
# Gating Network
# ==========================================
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 20),    # Hidden layer for better decision making
            nn.ReLU(),
            nn.Linear(20, num_experts),
            nn.Softmax(dim=1)            # Ensure sum of probabilities = 1
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# Mixture of Experts (MoE) Architecture
# ==========================================
class MoE(nn.Module):
    def __init__(self, num_experts=2, input_dim=1, output_dim=1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        
        # Initialize Experts
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        
        # Initialize Gating Network
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # 1. Get outputs from all experts
        # Shape: [Batch, Num_Experts, Output_Dim]
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 2. Get probabilities from Gating Network
        # Shape: [Batch, Num_Experts]
        gate_probs = self.gate(x)

        return expert_outputs, gate_probs

# ==========================================
# Competitive Loss Function (Eq 1.3)
# ==========================================
class MoELoss(nn.Module):
    def forward(self, expert_outputs, gate_probs, target):
        # Expand target to match expert_outputs shape
        target_expanded = target.unsqueeze(1).expand_as(expert_outputs)

        # 1. Squared Error Calculation
        squared_error = (target_expanded - expert_outputs).pow(2).sum(dim=2)

        # 2. Convert to Gaussian Likelihood
        # Higher likelihood means smaller error
        likelihood = torch.exp(-0.5 * squared_error)

        # 3. Weighted Sum of Likelihoods
        weighted_likelihood = (gate_probs * likelihood).sum(dim=1)

        # 4. Negative Log-Likelihood (Minimize this)
        loss = -torch.log(weighted_likelihood + 1e-8)

        return loss.mean()
