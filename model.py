import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=64, n_heads=2, n_layers=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embed))
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embed, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        token_embeddings = self.token_embed(idx)  # (B, T, C)
        x = token_embeddings + self.pos_embed[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        size = x.size(1)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.ff(x))
        return x
