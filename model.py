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


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # the dimension of the embedding must be even
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # builds the theta parameters, shape: (head_dim / 2)
    # according to the formula theta_i = 10000 ^ (-2(i-1) / dim) for i = [1, 2, ... dim / 2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape; (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the position (the "m" parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using the outer product
    # shape: (seq_len) outer_product* (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # we can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch, seq_len, h, head_dim) -> (batch, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (batch, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (batch, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (batch, seq_len, h, head_dim / 2) -> (batch, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch, seq_len, h, head_dim / 2, 2) -> (batch, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
     
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensore):
        # (batch, seq_len, dim) * (batch, seq_len, 1) = (batch, seq_len, dim))
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pw(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (dim) * (batch, seq_len, dim) = (batch, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple of parameter
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModuleArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block 
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) --> (batch, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        output = h + self.feed_forward.forward(self.ffn_norm(h))
        return output


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
        
        def forward(self, tokens: torch.Tensor, start_pos: int):
            # (Batch, seq_len)
            batch_size, seq_len = tokens.shape
            assert seq_len ==1, "Only one token at a time can be processed"

            # (Batch, seq_len) -> (Batch, seq_len, Dim)
            h = self.tok_embeddings(tokens)

            # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
            freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

            # Consecutively apply all the encoder layers
            for layer in self.layers:
                    h = layer(h, start_pos, freqs_complex)
            h = self.norm(h)
            output = self.output(h).float()
            return output