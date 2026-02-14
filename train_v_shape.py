import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from src.model import MoE, MoELoss  # Import from local module

# ==========================================
# 1. Generate Data (V-Shape)
# ==========================================
def generate_v_data(n_samples=1000):
    # X values between -5 and 5
    x = torch.linspace(-5, 5, n_samples).unsqueeze(1)
    
    # Y values: Absolute value function (|x|) with noise
    # Left slope: -1, Right slope: +1
    y = torch.abs(x) + torch.randn(x.size()) * 0.1
    
    return x, y

x_train, y_train = generate_v_data()

# ==========================================
# 2. Initialize Model & Optimizer
# ==========================================
# We use 2 experts for V-shape (one for left, one for right)
model = MoE(num_experts=2)
criterion = MoELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# ==========================================
# 3. Training Loop
# ==========================================
print("Training Started (V-Shape)...")
epochs = 500

for epoch in range(epochs):
    # Forward pass
    expert_outputs, gate_probs = model(x_train)
    
    # Calculate Loss (Competitive Learning)
    loss = criterion(expert_outputs, gate_probs, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training Complete!")

# ==========================================
# 4. Visualization
# ==========================================
with torch.no_grad():
    expert_outputs, gate_probs = model(x_train)
    
    # Final Prediction (Weighted Average for visualization)
    final_pred = (expert_outputs * gate_probs.unsqueeze(2)).sum(dim=1)
    
    # Hard assignment: Choose the expert with the highest probability
    expert_choice = torch.argmax(gate_probs, dim=1)

plt.figure(figsize=(10, 5))

# Plot Original Data
plt.scatter(x_train, y_train, s=5, c='gray', alpha=0.5, label='Original Data')

# Plot Expert A Zone
mask_0 = expert_choice == 0
plt.scatter(x_train[mask_0], final_pred[mask_0], s=10, c='red', label='Expert A Zone')

# Plot Expert B Zone
mask_1 = expert_choice == 1
plt.scatter(x_train[mask_1], final_pred[mask_1], s=10, c='blue', label='Expert B Zone')

plt.title("Mixture of Experts: Task Decomposition (V-Shape)")
plt.legend()
plt.show()
