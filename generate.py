import torch
from model import MiniGPT
from tokenizer import CharTokenizer

def generate(model, tokenizer, start_text, length=100):
    model.eval()
    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long)
    for _ in range(length):
        logits = model(idx[:, -model.block_size:])
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())
