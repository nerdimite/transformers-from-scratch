import torch
from transformer import Transformer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SOS = 1, EOS = 2, PAD = 0
    X = torch.tensor([
        [1, 4, 6, 8, 10, 2, 0, 0],
        [1, 4, 8, 10, 12, 14, 16, 2]
    ]).to(device)
    y = torch.tensor([
        [1, 3, 5, 7, 9, 2, 0, 0],
        [1, 3, 7, 9, 11, 13, 15, 2]
    ]).to(device)

    pad_idx = 0
    src_vocab_size = 17
    trg_vocab_size = 16

    model = Transformer(src_vocab_size, trg_vocab_size, pad_idx, pad_idx).to(device)

    out = model(X, y[:, :-1])
    print(out.shape)