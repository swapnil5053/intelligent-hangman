import torch
import torch.nn as nn
import math

# Vocabulary Configuration
ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
VOCAB = {char: i for i, char in enumerate(ALPHABET)}
VOCAB['_'] = 26
VOCAB['<pad>'] = 27
VOCAB['<sep>'] = 28

VOCAB_SIZE = len(VOCAB)
REV_VOCAB = {v: k for k, v in VOCAB.items()}

# Model hyperparameters
MAX_WORD_LEN = 20
SEQ_LEN = MAX_WORD_LEN + 1 + 26  # 20 word chars, 1 separator, 26 wrong guesses


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class HangmanTransformer(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 512, max_len: int = SEQ_LEN, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=VOCAB['<pad>'])
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        # Pad mask for transformer
        key_padding_mask = (x == VOCAB['<pad>'])
        
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        logits = self.decoder(encoded)  # [batch_size, seq_len, vocab_size]
        return logits
