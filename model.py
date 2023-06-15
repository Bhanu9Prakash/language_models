"""
model.py
======================
This module defines various components used to construct a Transformer-based language model. The module provides classes for the following elements:

1. `Head`: Defines a single head of attention in the Transformer mechanism.

2. `MultiHeadAttention`: Implements the Multi-Head Attention mechanism which is a key component of the Transformer architecture.

3. `FeedForward`: Defines the Feed Forward Network used inside each Transformer block.

4. `DecoderBlock`: Defines a single decoder block in the Transformer architecture. Each decoder block consists of a Multi-Head Attention layer followed by a Feed Forward Network.

5. `LanguageModel`: Implements the overall Language model which employs Decoder Blocks to transform input sequences. The Language Model includes token and positional embeddings and also provides functionality for token generation.

Each of these classes utilizes PyTorch's nn.Module and includes forward methods for the forward pass computation.

Classes
-------
Head : torch.nn.Module
    A single Head of attention.
MultiHeadAttention : torch.nn.Module
    Multi-Head Attention mechanism.
FeedForward : torch.nn.Module
    Feed Forward Network inside the Transformer.
DecoderBlock : torch.nn.Module
    A single Decoder block of the Transformer.
LanguageModel : torch.nn.Module
    Language model which utilizes the Decoder Blocks.
"""
#importing necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
  """
  This class represents a single Head of attention.

  Args:
      d_model: The dimensionality of the input and output of the head.
      d_head: The dimensionality of the internal workings of the head.
      dropout: dropout of the calculated scaled dot product
      context_length: context length of the model
  """
  def __init__(self, d_model:int, d_head:int, dropout:float, context_length:int) -> None:
    super().__init__()
    self.query = nn.Linear(in_features = d_model, out_features = d_head, bias = False)
    self.key = nn.Linear(in_features = d_model, out_features = d_head, bias = False)
    self.value = nn.Linear(in_features = d_model, out_features = d_head, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones((context_length, context_length))))
    self.dropout = nn.Dropout(dropout)
    self.d_head = d_head

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    """
    Forward pass for single Head attention.

    Args:
        x: Input tensor (dims:batch_size, input_length, d_model)
    Returns:
        wei: Output tensor after applying attention (dims:batch_size, input_length, d_head)
    """
    B, T, C = x.shape # Batch_Size, Input_Length, Dimension where C = d_model
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # dim : B,T,T
    wei = F.softmax(wei, dim = -1) / (self.d_head**0.5)
    out = wei @ v # dim : B, T, d_head
    return out

class MultiHeadAttention(nn.Module):
  """
  This class represents the Multi-Head Attention mechanism.

  Args:
    n_heads: The number of heads in the Multi-Head Attention.
    d_model: The dimensionality of the input and output of the head.
    d_head: The dimensionality of the internal workings of the head.
    attn_dropout: dropout of the calculated scaled dot product
    proj_dropout: dropout of the projection layer
    context_length: context length of the model
    """
  def __init__(self, n_heads:int, d_model:int, d_head:int, attn_dropout:float, proj_dropout:float, context_length:int) -> None:
    super().__init__()
    self.multihead = nn.ModuleList([Head(d_model = d_model, d_head = d_head, dropout = attn_dropout, context_length = context_length) for _ in range(n_heads)])
    self.proj = nn.Linear(in_features = d_model, out_features = d_model)
    self.dropout = nn.Dropout(proj_dropout)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    """
    Forward pass for Multi-Head Attention.

    Args:
        x: Input tensor.
    Returns:
        x: Output tensor after applying multi-head attention.
    """
    x = torch.cat([head(x) for head in self.multihead], dim = -1)
    x = self.dropout(self.proj(x))
    return x

class FeedForward(nn.Module):
  """
  This class represents the Feed Forward Network inside the transformer.

  Args:
      d_model: The dimensionality of the input and output of the feed forward network.
      d_hidden: The dimensionality of the internal workings of the feed forward network.
      dropout: dropout of the second feedforward layer
  """
  def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
    super().__init__()
    self.net = nn.Sequential(nn.Linear(in_features = d_model, out_features = d_hidden),
                             nn.ReLU(),
                             nn.Linear(in_features = d_hidden, out_features = d_model))

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    """
    Forward pass for Feed Forward Network.

    Args:
        x: Input tensor.
    Returns:
        Output tensor after applying feed forward network.
    """
    return self.net(x)

class DecoderBlock(nn.Module):
  """
  This class represents a single Decoder block of the Transformer.

  Args:
    d_model: The dimensionality of the input and output of the decoder block. (Embedding dimension)
    n_heads: The number of heads in the Multi-Head Attention mechanism.
    d_head: The dimensionality of the internal workings of the head.
    d_hidden: The dimensionality of the internal workings of the feed forward network.
    attn_dropout: dropout of the calculated scaled dot product
    proj_dropout: dropout of the projection layer
    ffwd_dropout: dropout of the feedforward block
    context_length: context length of the model
  """
  def __init__(self, d_model: int, n_heads: int, d_head: int, d_hidden: int, attn_dropout: float, proj_dropout: float, ffwd_dropout: float, context_length:int) -> None:
    super().__init__()
    self.self_attn = MultiHeadAttention(n_heads = n_heads, d_model = d_model, d_head = d_head, attn_dropout = attn_dropout, proj_dropout = proj_dropout, context_length = context_length)
    self.ffwd = FeedForward(d_model=d_model, d_hidden = d_hidden, dropout = ffwd_dropout)
    self.ln1 = nn.LayerNorm(normalized_shape = d_model)
    self.ln2 = nn.LayerNorm(normalized_shape = d_model)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    """
    Forward pass for Decoder Block.

    Args:
        x: Input tensor.
    Returns:
        x: Output tensor after applying decoder block.
    """
    x = x + self.self_attn(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class LanguageModel(nn.Module):
  """
  This class represents the Language model which utilizes the Decoder Blocks.

  Args:
      vocab_size: The number of unique tokens in the vocabulary.
      d_model: The dimensionality of the embeddings.
      context_length: The maximum length of the sequences.
      n_layer: The number of decoder blocks in the transformer.
      n_heads: The number of heads in the Multi-Head Attention mechanism.
      d_head: The dimensionality of the internal workings of the head.
      d_hidden: The dimensionality of the internal workings of the feed forward network.
      attn_dropout: dropout of the calculated scaled dot product
      proj_dropout: dropout of the projection layer
      ffwd_dropout: dropout of the feedforward block
  """
  def __init__(self, vocab_size: int, d_model: int,
               n_layer: int, n_heads: int, d_head: int, d_hidden: int,
               attn_dropout: float, proj_dropout: float, ffwd_dropout: float,
               context_length:int) -> None:
    super().__init__()
    # layers
    self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim = d_model)
    self.position_embedding_table = nn.Embedding(num_embeddings = context_length, embedding_dim = d_model)
    self.blocks = nn.Sequential(*[DecoderBlock(d_model = d_model, n_heads = n_heads, d_head = d_head, d_hidden = d_hidden, attn_dropout = attn_dropout, proj_dropout = proj_dropout, ffwd_dropout = ffwd_dropout, context_length = context_length) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(normalized_shape = d_model)
    self.lm_head = nn.Linear(in_features = d_model, out_features = vocab_size)
    self.context_length = context_length




  def forward(self, idx : torch.Tensor) -> torch.Tensor:
    """
    Forward pass for Language Model.

    Args:
        idx: Input tensor of indices. (dims:batch_size, input_length)
        targets: Target tensor of indices. (dims:batch_size, input_length)
    Returns:
        logits: Output tensor of logits. (dims:batch_size, input_length, vocab_size)
    """
    B, T = idx.shape # batch_size, input_length
    token_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T))
    x = token_emb + pos_emb
    logits = self.lm_head(self.ln_f(self.blocks(x)))

    return logits

  def generate(self, idx:torch.Tensor, max_new_tokens:int = 50) -> torch.Tensor:
    """
    Function to generate new tokens.

    Args:
        idx: Input tensor of indices.
        max_new_tokens: The maximum number of new tokens to be generated. Defaults to 50.
    Returns:
        idx: Output tensor after generating new tokens.
    """
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.context_length:]
      logits = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = -1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat([idx, idx_next], dim = -1)
    return idx
