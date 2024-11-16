import torch
import torch.nn as nn
import math
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        logging.info("Initializing PositionalEncoding...")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)    # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # Apply cos to odd indices
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, embed_size]
        self.register_buffer('pe', pe)
        logging.info("PositionalEncoding initialized.")

    def forward(self, x):
        # x shape: [seq_length, batch_size, embed_size]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, hidden_dim=2048, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        logging.info("Initializing TransformerModel...")
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=False  # Ensure batch_first is False for consistency
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        logging.info("TransformerModel initialized.")

    def _init_weights(self):
        logging.info("Initializing weights...")
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        logging.info("Weights initialized.")

    def generate_square_subsequent_mask(self, sz):
        # Generates an upper-triangular matrix of True values, with zeros on the diagonal.
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, src_mask=None):
        # src shape: [seq_length, batch_size]
        logging.debug(f"Input shape: {src.shape}")
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        if src_mask is None:
            device = src.device
            seq_len = src.size(0)
            src_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        output = self.transformer_encoder(src, mask=src_mask)
        logits = self.fc_out(output)
        logging.debug(f"Output shape: {logits.shape}")
        return logits  # Shape: [seq_length, batch_size, vocab_size]

if __name__ == "__main__":
    logging.info("Testing TransformerModel...")
    # Example parameters
    vocab_size = 10000
    embed_size = 512
    num_heads = 8
    hidden_dim = 2048
    num_layers = 6
    dropout = 0.1

    # Initialize the model
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Create dummy input data
    seq_length = 128
    batch_size = 32
    dummy_input = torch.randint(0, vocab_size, (seq_length, batch_size))
    logging.info("Dummy input created.")

    # Forward pass
    output = model(dummy_input)
    logging.info("Forward pass completed.")
    logging.info(f"Model output shape: {output.shape}")