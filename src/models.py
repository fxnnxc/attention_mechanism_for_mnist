

import torch 
import torch.nn as nn

class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 32 
        num_heads = 4 
        self.patch_encoder = nn.Linear(16, hidden_dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 10)
        self.query = nn.Parameter(torch.rand(1,1,hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_encoder(x)
        query = self.query.repeat(batch_size, 1,1)
        attn_output, attn_output_weights = self.self_attention(query=query, key=x, value=x)
        x = attn_output[:,0,:]
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = AttentionPool()
    x = torch.rand(32, 49, 16)
    y= model.forward(x)
    print(y.size())
