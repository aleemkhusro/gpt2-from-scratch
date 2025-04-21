from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import get_batch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_interval = 20
eval_iters = 200
learning_rate = 3e-4

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd:int = 384
    batch_size: int = 64    

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ff1 = nn.Linear(config.n_embd, config.n_embd *4)
        self.ff2 = nn.Linear(config.n_embd*4, config.n_embd)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.gelu(self.ff1(x), approximate='tanh')
        x= self.dropout(x)
        x = self.ff2(x)
        x = self.dropout(x)
        return x
    

class CausalSelfAttention(nn.Module):
    #this is the batched implementation
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.main_weight_matrix = nn.Linear(config.n_embd, config.n_embd*3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.2)
        self.resid_dropout = nn.Dropout(0.2)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.main_weight_matrix(x)
        q,k,v = torch.split(qkv,C, dim=-1)
        q = q.view(B,T,self.n_head,C//self.n_head) #B,T,nh,hs
        q = q.transpose(1,2) #B,nh,T,hs
        k = k.view(B,T,self.n_head,C//self.n_head) #B,T,nh,hs
        k = k.transpose(1,2) #B,nh,T,hs
        v = v.view(B,T,self.n_head,C//self.n_head) #B,T,nh,hs
        v = v.transpose(1,2) #B,nh,T,hs
        hs = (C//self.n_head)

        attn = (q@k.transpose(-2,-1)) * (hs**-0.5) #B,nh,T,T
        mask = self.tril[:T,:T].view(1,1,T,T)
        attn = attn.masked_fill(mask ==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        embd = attn @ v #B,nh,T,hs

        embd = embd.transpose(1,2) #B,T,nh,hs
        output = embd.contiguous().view(B,T,C)
        output = self.resid_dropout(output)
        output = self.proj(output)
        
        return output


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x +self.mlp(self.ln2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # ── embeddings ─────────────────────────────────
        self.emb_drop = nn.Dropout(0.2)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        # ── language‑model head ───────────────────────
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying scheme
        self.lm_head.weight = self.transformer['wte'].weight
    
    def forward (self, idx, target=None):
        B,T = idx.shape
        tok_embd = self.transformer['wte'](idx) #B,T,C
        pos_ids = torch.arange(T, device=device).unsqueeze(0) #1,T
        pos_embd = self.transformer['wpe'](pos_ids) #1,T,C
        x = tok_embd+pos_embd #B,T,C
        x = self.emb_drop(x)
        for block in self.transformer['h']:
            x = block(x)
        logits = self.lm_head(self.transformer['ln_f'](x))
        if target is not None:
            B,T,C = logits.shape
            target = target.view(B*T)
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, target)
            return logits, loss
        
        return logits, None

def evaluate_loss(model: nn.Module, config: GPTConfig):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        
        
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def evaluate_loss_overfit(model: nn.Module, config: GPTConfig, X, Y):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        
        
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            logits, loss = model(X,Y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

config = GPTConfig()

model = GPT(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

max_iters = 1000
xb, yb = get_batch('train', config)
for iteration in range(max_iters):
    # if iteration % eval_interval ==0 and iteration >0:
    #     losses_records = evaluate_loss_overfit(model, config, xb, yb)
    #     print(f"step {iter}: train loss: {losses_records['train']:.4f}, val loss: {losses_records['val']:.4f}")

    optimizer.zero_grad()
    _, loss = model(xb, yb)
    loss.backward()
    with torch.no_grad():
        print(loss.mean())
    optimizer.step()






    

