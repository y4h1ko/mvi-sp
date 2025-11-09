from .imports_and_libraries import *
from .positional_encodings import *


#e-only transformer, head (w)
class TransformerModel1(nn.Module):
    def __init__(self, seq_len: int=1, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_f: int = 128, dropout: float = 0.1):
        super().__init__()

        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)       #creates order for time
        
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))     #normalization and prediction for w

    def forward(self, src):
        src = src.unsqueeze(-1)

        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        output = self.head(pool)
        return output
