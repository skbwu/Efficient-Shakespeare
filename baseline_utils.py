# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import multinomial
import time; import pandas as pd

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import requests
import os
import pdb

torch.manual_seed(305)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global hyperparameters
SMALL_ITERS = 2000
LARGE_ITERS = 2000
EVAL_ITERS = 100
CONTEXT_WINDOW_SIZE = 256


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
    
    
class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6, n_layers=6):
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