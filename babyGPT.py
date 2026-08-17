import numpy as np
import math
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
block_size = 256
n_embd = 768
n_heads = 12
n_block = 12

text = ''
with open('BabyGPT-Quantization/shakespeare.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: idx for idx, ch in enumerate(chars)}
itos = {idx: ch for idx, ch in enumerate(chars)}

#hdsjkfhaios
# print (stoi, itos)
# {'a': 0, 'd': 1, 'f': 2, 'h': 3, 'i': 4, 'j': 5, 'k': 6, 'o': 7, 's': 8} {0: 'a', 1: 'd', 2: 'f', 3: 'h', 4: 'i', 5: 'j', 6: 'k', 7: 'o', 8: 's'}

encode = lambda x: [stoi[i] for i in x]
decode = lambda x: "".join(itos[i] for i in x)

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.layer_norm(x)

class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_expansion = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_expansion(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim = -1)
        q = q.view(B, T, n_heads, C // n_heads).transpose(1, 2)
        k = k.view(B, T, n_heads, C // n_heads).transpose(1, 2)
        v = v.view(B, T, n_heads, C // n_heads).transpose(1, 2)
        scores = q @ k.transpose(-1, -2)
        scores = scores / (math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        scores = self.softmax(scores)
        y = scores @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln1 = LayerNorm()
        self.ln2 = LayerNorm()
        self.mlp = MLP()
        self.attention = Attention()

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = nn.ModuleList([Block() for _ in range(n_block)])
        self.softmax = nn.Softmax()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ln_f = LayerNorm()

    def forward(self, idx, targets = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype = torch.long, device=idx.device)
        idx = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            idx = block(idx)
        idx = self.ln_f(idx)
        logits = self.lm_head(idx)

        if targets is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_token, temperature = 1.0):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = self.softmax(logits)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim = -1)
        return idx

model = GPT().to(device)
lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

iters = 5000
for i in range(iters):
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    if (i + 1) % 1000 == 0: 
        print (f"iter{i + 1}, Loss: {loss:.4f}")

context = torch.tensor([encode("What should I do")], dtype = torch.long, device = device)
gen = model.generate(context, max_new_token = 500, temperature = 0.8)
print(decode(gen[0].tolist()))
torch.save(model.state_dict(), "BabyGPT-Quantization/checkpoint.pt")