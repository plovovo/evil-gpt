import torch

def prepare_data(text, tokenizer, block_size=64):
    data = tokenizer.encode(text)
    x, y = [], []
    for i in range(len(data) - block_size):
        x.append(data[i:i+block_size])
        y.append(data[i+1:i+1+block_size])
    return torch.tensor(x), torch.tensor(y)
