import torch
from torch import nn
from torch.nn.modules.activation import ReLU

from self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention_layer = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, keys, values, queries, mask):
        
        # Multi-Headed Attention and Add-Norm
        attention = self.attention_layer(keys, values, queries, mask)
        x = self.dropout(self.norm1(attention + queries))

        # Feedforward and Add-Norm
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, 
                    heads, 
                    dropout, 
                    forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

        