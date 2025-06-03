import tiktoken
import torch 

class DataLoaderLite:
    def __init__(self,ddp_rank, ddp_world_size, B, T):
        self.B = B
        self.T = T
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # at init load tokens from disk and store them in memory
        with open('Data/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = B * T * ddp_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor. Multiply by world size because that's the number of processes we have in ddp
        self.current_position += B * T * self.ddp_world_size
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.ddp_world_size + 1) > len(self.tokens):
            self.current_position = B * T * self.ddp_rank
        return x, y