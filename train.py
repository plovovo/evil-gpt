import torch
import torch.nn.functional as F
from model import MiniGPT
from tokenizer import CharTokenizer
from data import prepare_data

with open("materny_gpt_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
x, y = prepare_data(text, tokenizer)

vocab_size = tokenizer.vocab_size
block_size = x.shape[1]
model = MiniGPT(vocab_size, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 100
batch_size = 32

for epoch in range(epochs):
    i = torch.randint(0, x.size(0), (batch_size,))
    xb, yb = x[i], y[i]
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
