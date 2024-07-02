import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from  dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32 # Number of heads for queries
    n_kv_heads: Optional[int] = None # Number of heads for the K and V
    vocab_size: int = -1 # Will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None 
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class Transformer(nn.Module): # Defines the whole of the model, except for Softmax
    
    def __init__(self, args: ModelArgs) -> None: 
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim) # Convert our input into embeddings

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers): # Pass our input embeddings through a list of layers
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps) # Last layer's output is sent through RMS Normalization
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False) # After RMSNorm, finally sent to the output layer 

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, 
                                                              self.args.max_seq_len * 2, device=self.args.device)
        