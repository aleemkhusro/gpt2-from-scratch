from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import DataLoaderLite
from utils.learning_rate_scheduler import get_lr
from utils.optimizers import configure_optimizers
from utils import gradient_accumulation
import sys
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_interval = 20
eval_iters = 200
learning_rate = 3e-4

@dataclass
class GPTConfig:
    block_size: int = 1024
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

# --------------------------- DDP settings

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc. This evaluates to True when the ddp_rank = 0
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda'
print( f" I am GPU {ddp_rank}, and the local rank is {ddp_local_rank}")

# --------------------------model init and train
import time
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')
config = GPTConfig()
model = GPT(config)
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module
optimizer = configure_optimizers(weight_decay=0.1, learning_rate = 6e-04, model = raw_model, master_process = master_process)

train_loader = DataLoaderLite(ddp_rank, ddp_world_size, B=16, T=1024 )
torch.set_float32_matmul_precision('high') #uses TFloat32 when mixed precision is used in autocast below. Rest everything is bfloat16. 

gradient_accum_steps = gradient_accumulation.get_step_count(train_loader.B , train_loader.T, ddp_world_size)

if master_process:
    print(f'gradient accumulation steps: {gradient_accum_steps}' )

for iteration in range(150):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    # accumulation needed to simulate large batch size to reprdouce the results of gpt2. Can't fit 0.5M tokens in one batch on consumer grade GPU.
    for micro_step in range(gradient_accum_steps):
        xb, yb = train_loader.next_batch()
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        #the loss is the mean of the loss of all batches. When using gradient accumulation, you also need to normalize with the number of accumulations to have proper mean.
        loss = loss / gradient_accum_steps
        loss_accum += loss.detach() #using loss.itme() will result in GPU->CPU trip every micro batch
        if ddp:
            # we dont need to sync the gradients between all the GPUs at every loss.backwards. We can allow for gradient accumulation and then sync the gradients at the last step
            steps_diff = gradient_accum_steps - micro_step
            if steps_diff == 1:
                model.require_backward_grad_sync = True
            else:
                model.require_backward_grad_sync = False
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, async_op=dist.ReduceOp.AVG)
    #compute L2 norm of all the gradient tensors viewed as a single vector and clip all gradients if the total norm exceed the threshold supplied here. Happens before optimizer step. 
    # this is a global norm based clip, and not a per-parameter-clip. max_norm as 1.0 is the setting in the gpt3 paper so thats where he got it from. 
    # Unlucky batch -> high loss -> high gradient update -> shock the model. So clip. 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for param_group in optimizer.param_groups:
        lr = get_lr(iteration)
        param_group['lr'] =lr #cosine decayed learning rate scheduler
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_processed = (train_loader.B * train_loader.T * gradient_accum_steps * ddp_world_size) 
    tokens_persec = tokens_processed/(t1-t0)
    if master_process:
        print(f"step {iteration} | loss: {loss_accum.item():.6f} | norm: {norm:.4f}  | lr = {lr:.4e} | dt: {dt:.2f}ms | tok/sec: {tokens_persec:.2f}")

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


        





    

