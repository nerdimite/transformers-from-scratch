import torch
from torch import nn

from self_attention import SelfAttention
from encoder import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()

        self.attention_layer = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, values, src_mask, trg_mask):

        # Masked Multi-Headed Attention and Add-Norm
        attention = self.attention_layer.forward(x, x, x, trg_mask)
        queries = self.dropout(self.norm(attention + x))

        # Multi-Headed Attention with Encoder Outputs
        # and Masked Multi-Headed Attention Outputs
        # with Add-Norm
        out = self.transformer_block(keys, values, queries, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_len,
        device,
        pos_embed=True
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        self.pos_embed = pos_embed
        if pos_embed:
            self.positional_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape

        if self.pos_embed:
            positions = torch.arange(0, seq_len).expand(
                N, seq_len).to(self.device)
        else:
            # const
            positions = torch.div(torch.arange(0, seq_len).expand(
                N, seq_len).unsqueeze(2).expand(N, seq_len, 256), 16).to(self.device)

        if self.pos_embed:
            out = self.dropout(self.word_embedding(
                x) + self.positional_embedding(positions))
        else:
            out = self.dropout(self.word_embedding(x) + positions)

        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(out)

        return out
