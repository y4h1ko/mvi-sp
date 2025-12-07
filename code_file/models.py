from .imports_and_libraries import *
from .positional_encodings import *


#e-only transformer, head (w)
class TransformerModel1(nn.Module):
    '''Transformer model for time-series prediction using only encoder layers.'''

    def __init__(self, seq_len: int=cfg.discr_of_time, d_model: int=cfg.dmodel, nhead: int=cfg.nhead, num_layers: int=cfg.num_layers, 
                 dim_f: int=cfg.dim_f, dropout: float=cfg.dropout):
        
        super().__init__()

        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)       #creates order for time
        
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))     #normalization and prediction for w

    def forward(self, src):
        '''Forward pass'''

        src = src.unsqueeze(-1)

        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        output = self.head(pool)
        return output



class HeadWithFlow1(nn.Module):
    '''Normalizing flow head for omega.'''

    def __init__(self, context_dim: int, hidden_features: int=cfg.flow_hidden_features, num_layers: int=cfg.flow_num_layers):
        super().__init__()

        self.context_dim  = context_dim
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.context_net = nn.Linear(context_dim, hidden_features)

        #MAF - creating matrixes which will transform distribution
        transform_list = []
        for _ in range(num_layers):
            transform_list.append(transforms.MaskedAffineAutoregressiveTransform(features=1, hidden_features=hidden_features, context_features=hidden_features))

        #chaining transformation sequence
        transform = transforms.CompositeTransform(transform_list)

        #wraps flow as object and keeps the information
        base_dist = distributions.StandardNormal(shape=[1])
        self.flow = flows.Flow(transform=transform, distribution=base_dist)

    def encode_context_head(self, context):
        return self.context_net(context)

    def log_prob(self, omega, context):

        ctx = self.encode_context_head(context)
        log_p = self.flow.log_prob(inputs=omega, context=ctx)
        return log_p

    def sample(self, context, num_samples: int):
        #function gives out mean, uncertainity and shape of distribution
        ctx = self.encode_context_head(context)  
    
        samples_bs1 = self.flow.sample(num_samples=num_samples, context=ctx)
        samples = samples_bs1.permute(1, 0, 2) 
        return samples



class TransformerModel2(nn.Module):
    '''Normalizing-flow head.
        - forward(x) returns the mean of samples
        - log_prob(x, y) for training with NLL
        - sample(x, S) for uncertainty (S samples)'''

    def __init__(self, seq_len: int=cfg.discr_of_time, d_model: int=cfg.dmodel, nhead: int=cfg.nhead, num_layers: int=cfg.num_layers,
                 dim_f: int=cfg.dim_f, dropout: float=cfg.dropout, flow_hidden_features: int=cfg.flow_hidden_features, flow_num_layers: int=cfg.flow_num_layers):

        super().__init__()

        self.input_embedding   = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)

        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.pre_head_norm = nn.LayerNorm(d_model)

        #flowhead
        self.flow_head = HeadWithFlow1( context_dim=d_model, hidden_features=flow_hidden_features, num_layers=flow_num_layers)


    def forward(self, src):
        '''Forward pass'''

        src = src.unsqueeze(-1)

        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)
        
        with torch.no_grad():
            samples = self.flow_head.sample(head_norm, num_samples=20)
        mu = samples.mean(dim=0)
        return mu

    def log_prob(self, src, target):
        src = src.unsqueeze(-1)

        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)
        
        log_p = self.flow_head.log_prob(target, context=head_norm)
        return log_p

    def sample(self, src, num_samples: int = 100):
        src = src.unsqueeze(-1)

        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool) 
        
        samples = self.flow_head.sample(head_norm, num_samples)
        return samples



class HeadWithFlow2w(nn.Module):
    '''Normalizing flow head for omegas - now mixture of 2.'''

    def __init__(self, context_dim: int, hidden_features: int=cfg.flow_hidden_features, num_layers: int=cfg.flow_num_layers, 
                 out_dim: int=2, use_permutations: bool=True):
        super().__init__()

        self.context_dim  = context_dim
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.context_net = nn.Linear(context_dim, hidden_features)

        #MAF - transform distribution with matrixes
        transform_list = []
        for i in range(num_layers):
            maf = transforms.MaskedAffineAutoregressiveTransform(features=out_dim, hidden_features=hidden_features, context_features=hidden_features)
            transform_list.append(maf)
            if use_permutations and i < num_layers - 1:
                perm = transforms.RandomPermutation(features=out_dim)
                transform_list.append(perm)

        #chaining transformation sequence
        transform = transforms.CompositeTransform(transform_list)

        #wraps flow as object and keeps the information
        base_dist = distributions.StandardNormal(shape=[out_dim])
        self.flow = flows.Flow(transform=transform, distribution=base_dist)

    def encode_context_head(self, context):
        return self.context_net(context)

    def log_prob(self, omega, context):
        ctx = self.encode_context_head(context)
        log_p = self.flow.log_prob(inputs=omega, context=ctx)
        return log_p

    def sample(self, context, num_samples: int):
        ctx = self.encode_context_head(context)  
    
        samples = self.flow.sample(num_samples=num_samples, context=ctx)
        if samples.dim() == 3:
            S0, S1, D = samples.shape

            if S0 == num_samples:
                samples = samples.permute(1, 0, 2)
                print('A')
            elif S1 == num_samples:
                samples = samples

        return samples



class TransformerModel3(nn.Module):
    '''Normalizing-flow head.
        - forward(x) returns the mean of samples
        - symetric log_prob(x, y) for training with NLL
        - sample(x, S) for uncertainty (S samples)'''

    def __init__(self, seq_len: int=cfg.discr_of_time, d_model: int=cfg.dmodel, nhead: int=cfg.nhead, num_layers: int=cfg.num_layers,
                 dim_f: int=cfg.dim_f, dropout: float=cfg.dropout, flow_hidden_features: int=cfg.flow_hidden_features, flow_num_layers: int=cfg.flow_num_layers,
                 out_dim: int=2):

        super().__init__()

        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)

        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.pre_head_norm = nn.LayerNorm(d_model)

        #flowhead
        self.flow_head = HeadWithFlow2w(context_dim=d_model, hidden_features=flow_hidden_features, num_layers=flow_num_layers, out_dim=out_dim)

    def forward(self, src, num_samples: int = 50):
        '''Forward pass'''
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)

        samples = self.flow_head.sample(head_norm, num_samples)
        samples_sorted, _ = torch.sort(samples, dim=-1)
        mu = samples_sorted.mean(dim=1)
        return mu

    def log_prob(self, src, target):
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)

        #(w1, w2) and (w2, w1) and their mixture
        log_p1 = self.flow_head.log_prob(target, context=head_norm)
        log_p2 = self.flow_head.log_prob(target.flip(dims=[1]), context=head_norm)

        #p_sym = 0.5 * (p(w1,w2) + p(w2,w1))
        stacked = torch.stack([log_p1, log_p2], dim=1)
        log_p_sym = torch.logsumexp(stacked, dim=1) - math.log(2.0)

        return log_p_sym

    def sample(self, src, num_samples: int=100):
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool) 
        
        samples = self.flow_head.sample(head_norm, num_samples)
        return samples