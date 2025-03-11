#!/usr/bin/env python
# coding: utf-8

# In[35]:


# torch imports
import numpy as np
import pandas as pd
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import multinomial

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import time

import requests
import os, re
import pdb
from gensim.models import Word2Vec

torch.manual_seed(305)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global hyperparameters
SMALL_ITERS = 2000
LARGE_ITERS = 2000
EVAL_ITERS = 100
CONTEXT_WINDOW_SIZE = 64 # used to be 256

# simulation settings
mc = [1, 3, 5][int(sys.argv[1])]

# load in the raw shakespeare, tokenize with words + special characters
with open("input.txt", "rt") as file:
    raw_corpus = file.readlines()
    
# special characters that we do want to keep!
spc = r'\w+|[^\w\s]|[\n\t\r\f\v]'

# get our corpus in splitted form for word2vec
corpus = [re.findall(spc, line) for line in raw_corpus]

# combine into one big list for aggregate statistics
combined = []
for line in corpus:
    combined += line

# load in our superset Word2Vec tokenizer (the biggest model)
biggest_model = Word2Vec.load("word2vec_models/mc=1_vs=1152.model")

# load in the Word2Vec tokenizer that we will be using (limited vocabulary)
w2v_model_limited = Word2Vec.load(f"word2vec_models/mc={mc}_vs=192.model")
vocab_size = len(w2v_model_limited.wv.key_to_index)

# get our data as WORDS - tokenize into word2vec
data = []
for word in tqdm(combined):
    if word in w2v_model_limited.wv.key_to_index:
        data.append(biggest_model.wv.key_to_index[word])
    else:
        
        # find the closest word
        closest_idx = np.argmax(
            [biggest_model.wv.similarity(word, reference_word) 
             for reference_word in w2v_model_limited.wv.index_to_key])
        data.append(closest_idx)
        
# convert everything into a tensor
data = torch.tensor(data, dtype=torch.long, device=device)

# split into a 90/10 train-test split
train_data, val_data = data[:int(len(data)*0.9)], data[int(len(data)*0.9):]

# function for getting batches of data (borrowed from HW)
def get_batch(split, context_window_size, device, batch_size=32):
    """
    generate a small batch of data of inputs x and targets y

    Args:
        split: 'train' or 'val'
        device: 'cpu' or 'cuda' (should be 'cuda' if available)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_window_size, (batch_size,))
    x = torch.stack([data[i:i+context_window_size] for i in ix])
    y = torch.stack([data[i+1:i+context_window_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

# helper function for tracking loss during training
@torch.no_grad()
def estimate_loss(model, eval_iters, context_window_size, device):
    """
    Args:
      model: model being evaluated
      eval_iters: number of batches to average over
      context_window_size: size of the context window
      device: 'cpu' or 'cuda' (should be 'cuda' if available)
    """
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, context_window_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


# In[18]:


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, context_window_size, embed_size=384):
        """
        Args:
          head_size: int, size of the head embedding dimension (K)
          context_window_size: int, number of tokens considered in the past for attention (T)
          embed_size: int, size of the token embedding dimension (D)
        """
        super().__init__()
        self.head_size = torch.tensor(head_size)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        # not a param of the model, so registered as a buffer
        self.register_buffer('tril', torch.tril(
            torch.ones(context_window_size, context_window_size)))

    def forward(self, x):
        """
        Args:
          x: (B,T,D) tensor of token embeddings

        Returns:
          (B,T,D) tensor of attention-weighted token embeddings
        """
        # TODO: your code here
        
        # 0. get the shape of x (will be important during inference)
        B, T, D = x.shape
        
        # 1. X U_q^T @ U_k X^T
        output = self.query(x) @ self.key(x).mT
        
        # 2. apply causal mask and divide by sqrt(K) - for inference later, we need to truncate this
        output = output.masked_fill(self.tril[:T,:T] == 0.0, float("-inf")) / (self.head_size ** 0.5)
        
        # 3. apply softmax-across-rows
        output = torch.softmax(output, dim=2)
        
        # 4. multiply by XV^T and return
        return output @ self.value(x)


# In[19]:


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, context_window_size, num_heads, head_size, embed_size=384):
        """
        Args:
            context_window_size: int, number of tokens considered in the past for attention (T)
            num_heads: int, number of heads (H)
            head_size: int, size of the head embedding dimension
            embed_size: int, size of the token embedding dimension
        """
        super().__init__()
        # TODO, your code below
        self.heads = nn.ModuleList(
            [Head(head_size, context_window_size, embed_size) for _ in range(num_heads)])

    def forward(self, x):
        # TODO, your code below
        
        # evaluate each head, pancake stack + sum
        output = torch.stack([head(x) for head in self.heads], axis=0).sum(axis=0)
        return output


# In[20]:


# run this cell to initialize this deep learning module that you should use in the code your write later
# you don't need to edit this layer
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity
        Given to you, you don't need to write any code here!
    """

    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)


# In[21]:


class TransformerBlock(nn.Module):
    """ Transformer block: communication across sequence length, followed by communication across embedding space
        Uses multi-headed attention
    """

    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        # TODO: your code below
        head_size = embed_size // num_heads
        self.feed_forward = FeedForward(embed_size)
        self.atten_heads = MultiHeadAttention(context_window_size, num_heads, head_size, embed_size)

    def forward(self, x):
        x = x + self.atten_heads(self.ln1(x)) # communication over sequence length
        x = x + self.feed_forward(self.ln2(x)) # communication across embedding space
        return x


# In[44]:


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_window_size, embed_size=384, 
                 num_heads=6, n_layers=6, freeze_type=True):
        """
          Args:
              vocab_size: int, number of tokens in the vocabulary (V)
              context_window_size: int, size of the context window (T)
              embed_size: int, embedding size (D)
              num_heads: int, number of heads (H)
              n_layers: int, number of layers (M)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window_size = context_window_size
        
        # remaining parameters fixed for now.
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(context_window_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(vocab_size,
                             context_window_size,
                             embed_size=embed_size,
                             num_heads=num_heads)
            for _ in range(n_layers)])

        # final layer norm
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

        # good initialization
        self.apply(self._init_weights)
        
        '''
        Major Change:
        1. Extract Embedding Vectors from Word2Vec, transplant into token_embedding_table.
        2. Potentially freeze or unfreeze the weights.
        '''
        if freeze_type is not None:
            pretrained_embeddings = torch.tensor(
                np.array([w2v_model.wv[word] for word 
                          in w2v_model.wv.key_to_index.keys()]))
            self.token_embedding_table = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_type)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, embed_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        """
        Agrgs:
            token_ids: tensor of integers, provides the contet, shape (B, T)
            targets: tensor of integers, provides the tokens we are preidcitng, shape (B, T)
        """
        B, T = token_ids.shape

        # token_ids and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(token_ids) # (B, T, D)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, D)
        x = tok_emb + pos_emb # (B, T, D)

        # TODO: your code below (copied from previous code cell)
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        
        # carbon copy (once again) from above
        if targets is None:
            loss = None
        else:
            # treat this as a classification problem - do cross entropy AVERAGED PER TOKEN!
            loss = F.cross_entropy(input=logits.reshape(-1, self.vocab_size), 
                                   target=targets.flatten()) # F.cross_entropy averages per token by default

        return logits, loss


    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        """
        Args:
            token_ids: tensor of integers forming the context, shape (B, T)
            max_new_tokens: int, max number of tokens to generate
        """
        # take in context, compute probabilities of next token, sample, repeat
        for t_pred in range(max_new_tokens):
            
            # get probability vectors given the current context
            logits, loss = self(token_ids[:,-self.context_window_size:])

            # sample our next token for each batch, augment to token_ids (our context)
            token_ids = torch.hstack(
                [token_ids, torch.multinomial(
                    input=torch.softmax(logits[:,-1,:], dim=1), num_samples=1)])
            
        # after we've finished generating our entire forecast horizon, output everything.
        return token_ids


# In[45]:


# settings to loop over
for embed_size in [192, 384, 576, 768, 960]:
    for freeze_type in [True, False, None]:

        # set a seed for "reproducibility"
        torch.manual_seed(4513215)
        
        # create a directory if necessary
        if "models" not in os.listdir():
            os.mkdir("models")
        foldername = f"mc={mc}_embed-size={embed_size}_freeze-type={str(freeze_type)}" 
        if foldername not in os.listdir("models"):
            os.mkdir(f"models/{foldername}")
        
        # status update
        print(f"Training model with embed_size={embed_size} and freeze_type={str(freeze_type)}.")
        
        # load in our word2vec embeddings.
        w2v_model = Word2Vec.load(f"word2vec_models/mc={mc}_vs={embed_size}.model")

        # initialize the model
        trans = TransformerLM(
            vocab_size=vocab_size, context_window_size=CONTEXT_WINDOW_SIZE,
            embed_size=embed_size, num_heads=6, n_layers=6, freeze_type=freeze_type)
        tlm = trans.to(device)

        # set a learning rate + optimizer
        learning_rate = 5e-5
        optimizer = torch.optim.AdamW(trans.parameters(), lr=learning_rate)
        eval_interval = 200

        loss_list, wallclock_list = [], []

        for it in tqdm(range(SMALL_ITERS)):

            # start our timer
            start = time.time()

            # every once in a while evaluate the loss on train and val sets
            if it % eval_interval == 0 or it == SMALL_ITERS - 1:
                print(f"iteration {it}")
                losses = estimate_loss(tlm, EVAL_ITERS, CONTEXT_WINDOW_SIZE, device)
                print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = get_batch('train', CONTEXT_WINDOW_SIZE, device)

            # evaluate the loss
            logits, loss = tlm(xb, yb)
            loss_list.append(loss.detach().item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # end our timer
            end = time.time()
            wallclock_list.append(end - start)

        # save our model weights at the end + also save our logs
        torch.save(tlm, f"models/{foldername}/model.pth")
        logs = pd.DataFrame(data={"loss" : loss_list, "wallclock" : wallclock_list})
        logs.to_csv(f"models/{foldername}/logs.csv", index=False)

        # clear cuda cache
        torch.cuda.empty_cache()