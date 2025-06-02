from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import DataLoaderLite
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_interval = 20
eval_iters = 200
learning_rate = 3e-4

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd:int = 768
    batch_size: int = 64
    use_flash_attn: bool = True    

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ff1 = nn.Linear(config.n_embd, config.n_embd *4)
        self.ff2 = nn.Linear(config.n_embd*4, config.n_embd)
        self.ff2.SCALE_INIT = 1 #flag for special inits
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
        self.config = config
        self.n_head = config.n_head
        self.main_weight_matrix = nn.Linear(config.n_embd, config.n_embd*3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.SCALE_INIT = 1 #add a flag to check for special initialization. Only proj weights scaled because it takes part in resdual connections. Cumulative adding 
        #can increase the variance of the activation because y = x + f(x) in each block, n_layer times. 
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
        if not self.config.use_flash_attn:
            attn = (q@k.transpose(-2,-1)) * (hs**-0.5) #B,nh,T,T
            mask = self.tril[:T,:T].view(1,1,T,T)
            attn = attn.masked_fill(mask ==0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            embd = attn @ v #B,nh,T,hs
        else:
            #using flash attention
            embd = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        embd = embd.transpose(1,2) #B,T,nh,hs
        output = embd.contiguous().view(B,T,C)
        output = self.proj(output)
        output = self.resid_dropout(output)
        
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

        # weight tying scheme. Inputs and outputs live in the same semantic space
        self.lm_head.weight = self.transformer['wte'].weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            #in the beginning the std is 0.02 as per the gpt2 paper.
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                #Factor of 2 is multiplied here because there are two residual connections, one for attn, and one for mlp. 
                std *= (2*self.config.n_layer) **-0.5
            torch.nn.init.normal_(module.weight, mean =0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, torch.nn.Embedding):
            #1/sqrt(dmodel) also comes out to ~0.02 for different dmodlels like 768, 1024 etc. This is what they chose in the gpt2 papers
            torch.nn.init.normal_(module.weight, mean=0, std =0.02)
    
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


# --------------------------model init and train
import time
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')
config = GPTConfig()
model = GPT(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

train_loader =DataLoaderLite(B=8, T=512)
# torch.set_float32_matmul_precision('high')

max_iters = 1
# xb, yb = get_batch('train', config)
for iteration in range(50):
    t0 = time.time()
    # if iteration % eval_interval ==0 and iteration >0:
    #     losses_records = evaluate_loss_overfit(model, config, xb, yb)
    #     print(f"step {iter}: train loss: {losses_records['train']:.4f}, val loss: {losses_records['val']:.4f}")
    xb, yb = train_loader.next_batch()
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(xb, yb)
    loss.backward()
    with torch.no_grad():
        print(loss.mean())
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_processed = (train_loader.B * train_loader.T) 
    tokens_persec = tokens_processed/dt
    print(f"step {iteration} | loss: {loss.item():.6f} | dt: {dt:.2f}ms | tok/sec: {tokens_persec:.2f}")

import sys
sys.exit(0)
# -------------------------------------- generate from the model 
num_return_sequences = 5
max_length = 100
model.eval()

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model and we are,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
x = tokens.to('cuda')


while x.size(1) < max_length:
    with torch.no_grad():
        logits,_ = model(x) # B,T,vocab_size
        logits = logits[:,-1,:] # get the logits of the last timestep
        probs = F.softmax(logits, dim=-1)
        #do top-k sampling where k=50
        #tok_prbs is B,50 and topk_indices is also B,50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) #B,1
        xcol = torch.gather (topk_indices, -1, ix) #B,1
        x= torch.cat((x,xcol), dim=1) #B,T+1

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


        





    

