import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])

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
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


import pandas as pd

def load_dataset_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = ""
    for _, row in df.iterrows():
        q, a = row["question"], row["answer"]
        text += f"Q: {q}\nA: {a}\n\n"
    return text


def prepare_data(text, tokenizer, block_size=64):
    data = tokenizer.encode(text)
    x, y = [], []
    for i in range(len(data) - block_size):
        x.append(data[i:i+block_size])
        y.append(data[i+1:i+1+block_size])
    return torch.tensor(x), torch.tensor(y)


def train_model(model, x, y, vocab_size, epochs=50, batch_size=32):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        losses = []
        for _ in range(100):  # 100 iterations per epoch
            i = torch.randint(0, x.size(0), (batch_size,))
            xb, yb = x[i], y[i]
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


def generate(model, tokenizer, start_text, length=200, temperature=1.0, top_k=10):
    model.eval()
    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long)
    for _ in range(length):
        logits = model(idx[:, -model.block_size:])[:, -1, :] / temperature
        if top_k is not None:
            top_logits, top_idx = torch.topk(logits, k=top_k)
            probs = torch.softmax(top_logits, dim=-1)
            next_id = top_idx.gather(-1, torch.multinomial(probs, 1))
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())



if __name__ == "__main__":

    csv_path = "/Users/eaambartsumyan/Desktop/angry_gpt/angry_gpt_full_dataset.csv"

    print("Загрузка датасета...")
    csv_path = "angry_gpt_full_dataset.csv"
    raw_text = load_dataset_csv(csv_path)
    tokenizer = CharTokenizer(raw_text)
    x, y = prepare_data(raw_text, tokenizer, block_size=64)

    print("Инициализация модели...")
    model = MiniGPT(vocab_size=tokenizer.vocab_size, block_size=64)

    print("Начинаем обучение...")
    train_model(model, x, y, tokenizer.vocab_size, epochs=50)

    print("Сохраняем модель...")
    torch.save(model.state_dict(), "materny_gpt.pth")

    print("Генерация примера...")
    sample = generate(model, tokenizer, start_text="Q: Почему я такой тупой?\nA:", length=200)
    print("\n=== Сгенерировано ===\n")
    print(sample)
