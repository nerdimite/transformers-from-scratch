from tqdm.notebook import tqdm
import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_len=100,
        pos_embed=True
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_len,
            device,
            pos_embed
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_len,
            device,
            pos_embed
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(
            1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape

        # Create a Triangular Mask for each training example
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)
                              ).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)

        return out

    def train(self, X, y, epochs=200, lr=0.0001, pad_idx=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        pbar = tqdm(total=epochs)
        for i in range(epochs):

            # shift the target to the left so it predicts the last token
            out = self(X, y[:, :-1])
            out = out.reshape(-1, out.shape[2])
            # print('Output:', out.shape)

            target = y[:, 1:].reshape(-1)  # shift the labels to the right
            # print('Target:', target.shape)

            loss = criterion(out, target)
            # print(f"Loss {loss.item()}")
            loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            optimizer.step()

            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)

    def predict(self, source, device):
        preds = [1]  # Intialize predictions with SOS
        max_gen_len = 15

        for i in range(max_gen_len):

            input_tensor = torch.tensor(preds).unsqueeze(0).to(device)
            # print('Input Tensor:', input_tensor)

            # Predict targets
            # with torch.no_grad():
            out = self(source, input_tensor)
            # print('Raw Output Shape:', out.shape)

            # Get the last word with highest probability
            word_idx = out.argmax(dim=-1)[:, -1].item()
            # print('Next Word Index:', word_idx)

            # Append to outputs
            preds.append(word_idx)

            if word_idx == 2:  # If token is EOS then stop predicting
                break

        return preds
