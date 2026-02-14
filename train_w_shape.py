import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from src.model import MoE, MoELoss

# ==========================================
# 1. Generate Data (W-Shape / Double V)
# ==========================================
def generate_w_data(n_samples=2000):
    x = torch.linspace(-4, 4, n_samples).unsqueeze(1)
    
    # W-Shape formula: y = | |x| - 2 |
    y = torch.abs(torch.abs(x) - 2)
    
    # Add noise
    y += torch.randn(x.size()) * 0.05
    
    return x, y

x_train, y_train = generate_w_data()

# ==========================================
# 2. Initialize Model (4 Experts)
# ==========================================
# We need 4 experts for W-shape (4 linear segments)
model = MoE(num_experts=4)
criterion = MoELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

# ==========================================
# 3. Training Loop
# ==========================================
print("Training Started (W-Shape)...")
epochs = 800

for epoch in range(epochs):
    expert_outputs, gate_probs = model(x_train)
    loss = criterion(expert_outputs, gate_probs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ==========================================
# 4. Visualization
# ==========================================
with torch.no_grad():
    expert_outputs, gate_probs = model(x_train)
    expert_choice = torch.argmax(gate_probs, dim=1)
    final_pred = (expert_outputs * gate_probs.unsqueeze(2)).sum(dim=1)

plt.figure(figsize=(12, 6))

plt.scatter(x_train, y_train, s=1, c='lightgray', label='Data')

colors = ['red', 'blue', 'green', 'orange']
labels = ['Expert A', 'Expert B', 'Expert C', 'Expert D']

for i in range(4):
    mask = expert_choice == i
    plt.scatter(x_train[mask], final_pred[mask], s=10, c=colors[i], label=f'{labels[i]} Zone')

plt.title("Mixture of Experts: Task Decomposition (W-Shape)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print Gating Probabilities for specific points
print("\n[Expert Responsibility Check]")
test_points = [-3, -1, 1, 3]
names = ["Left Down", "Left Up", "Right Down", "Right Up"]

with torch.no_grad():
    test_tensor = torch.tensor(test_points).unsqueeze(1).float()
    _, probs = model(test_tensor)
    
    for name, point, p in zip(names, test_points, probs):
        winner = torch.argmax(p).item()
        print(f"X={point:>2} ({name}): Assigned to Expert {winner} (Prob: {p[winner]*100:.1f}%)")
