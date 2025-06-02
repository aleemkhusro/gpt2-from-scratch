import math
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    #   math.cos(π * decay_ratio) goes from cos(0) = 1 → cos(π) = –1
    #   so (1 + cos(...)) / 2 goes from 1 → 0 as decay_ratio goes 0 → 1
    return min_lr + coeff * (max_lr - min_lr) # gradually fall to min_lr as the coeff cosine decays from 1 to 0