
import torch
torch.manual_seed(1337)
# read it in to inspect it
with open('Data\\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#create train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% is train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split, config):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    xappender = []
    yappender = []
    for i in ix:
        xappender.append(data[i:i+config.block_size])
    x = torch.stack(xappender)
    for i in ix:
        yappender.append(data[i+1:i+config.block_size+1])
    y = torch.stack(yappender)
    x,y = x.to('cuda'), y.to('cuda')


    return x, y