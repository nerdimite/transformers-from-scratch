import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by the number of heads"

        # Weight matrices for generating Q, V, K
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # FC Output for Concatenating the outputs
        self.fc_out = nn.Linear(self.heads * self.head_dim, embed_size)

    def forward(self, keys, values, queries, mask=None):
        # Get batch size
        N = queries.shape[0]

        # Get source/target lengths
        key_len, value_len, query_len = keys.shape[1], values.shape[1], queries.shape[1]

        # Splitting the embeddings into chunks for head input
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Project Q, K, V using the parameter matrices
        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # Perform dot product of query and keys (energy)
        # query shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # => energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        # Mask the energy using the attention mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Get the scaled dot product attention
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # attention shape: (N, heads, query_len, key_len) [ value_len == key_len ]
        # values shape: (N, value_len, heads, head_dim)
        # => out shape: (N, query_len, heads, head_dim) = (N, query_len, heads * head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out


