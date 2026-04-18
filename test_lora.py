import torch
import torch.nn as nn
import sys
sys.path.append("nanoGPT")
from model_lora import CausalSelfAttention, GPTConfig, GPT


# Create a dummy config for testing
config = GPTConfig(
    vocab_size=5000,
    n_embd=256,
    n_layer=12,
    n_head=8,
    block_size=128,
    lora_rank=4  # Set LoRA rank
)

# Instantiate the model
model = CausalSelfAttention(config)

# Create dummy input (batch size 2, sequence length 10)
dummy_input = torch.randn(2, 10, config.n_embd)

# Perform a forward pass
output = model(dummy_input)
print("Output shape:", output.shape)  # Expected output shape: (2, 10, 256)

# Check if LoRA matrices are trainable
print("A matrix trainable:", model.q_proj.A.requires_grad)
print("B matrix trainable:", model.q_proj.B.requires_grad)

# Check if original weights are frozen (e.g., q_proj original weights)
print("Original q_proj weight frozen:", not model.q_proj.original.weight.requires_grad)

# Setup optimizer and perform one step (dummy loss)
params_to_update = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(params_to_update, lr=5e-5)

optimizer.zero_grad()
loss = output.sum()  # Dummy loss for testing
loss.backward()
optimizer.step()

# Ensure LoRA matrices (A and B) are updated
print("Updated A matrix:", model.q_proj.A.grad)
print("Updated B matrix:", model.q_proj.B.grad)

model = GPT(config)
dummy_tokens = torch.randint(0, config.vocab_size, (2, 16))
logits, loss = model(dummy_tokens)
print("Model logits shape:", logits.shape)
