from .imports_and_libraries import *


#positional encoding (copy of pytorch tutorial)
#anti-permutation - for model to know order in time
class PositionalEncoding(nn.Module):
    '''Standard positional encoding class from PyTorch tutorial'''

    def __init__(self, d_model: int=cfg.dmodel, max_len: int=cfg.discr_of_time):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)



