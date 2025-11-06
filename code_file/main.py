import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

SEED = 177013               #seed for reproducibility
NUM_OF_SAMPLES = 1000       #number of samples in dataset
TIME_INTERVAL = [0.0, 2*np.pi]      #interval for time
DISCR_OF_TIME = 10          #level of time discretization
OMEGA_MIN = 0               #minimal value for frequency (omega)
OMEGA_MAX = 10              #maximal value for frequency (omega)
AMPLITUDE_MIN = 1           #minimal value for amplitude (A)
AMPLITUDE_MAX = 1           #maximal value for amplitude (A)

EPOCHS = 200 


#sine dataset generator
def make_sine_dataset(N: int=NUM_OF_SAMPLES, t_interval: list=TIME_INTERVAL, t_k: int=DISCR_OF_TIME, w_min: int=OMEGA_MIN, w_max: int=OMEGA_MAX, seed=SEED ):
    rng = np.random.default_rng(seed)

    #evenly space time in t_k levels
    t_val = np.linspace(t_interval[0], t_interval[1], t_k)
    #generate N frequencies from [w_min ,w_max]
    w_val = rng.uniform(w_min, w_max, size=N)
    
    V, target = [], []

    #generate N samples of A*sin(w*t)
    for wi in w_val:
        Vi = 1 * np.sin(wi * t_val)
        V.append(np.array(Vi))
        target.append(np.array([wi]))

    #return tuple of arrays of 
    return np.array(V), np.array(target), t_val


V_np, tar_np, t_np = make_sine_dataset()

#np.arrays to tensors
X = torch.from_numpy(V_np).float()
y = torch.from_numpy(tar_np).float()
ds_full = TensorDataset(X, y)

#spltting to train, val and test parts
ds_train, ds_val, ds_test = random_split(
    ds_full, [0.6, 0.2, 0.2],
    generator=torch.Generator().manual_seed(0)
)

batch = min(32, max(1, len(ds_train)))
train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False)
test_loader = DataLoader(ds_test, batch_size=batch, shuffle=False)

#positional encoding (copy of pytorch tutorial)
#anti-permutation - for model to know order in time
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")

#setup
model = TransformerModel1(seq_len=DISCR_OF_TIME, d_model=128, nhead=4, num_layers=2, dim_f=256, dropout=0.1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    mse, mae, n = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        mse += torch.sum((pred - yb)**2).item()
        mae += torch.sum(torch.abs(pred - yb)).item()
        n += yb.numel()
    
    return mse / n, mae / n

@torch.no_grad()
def prediction_collecter_plot(loader, model, device):
    model.eval()
    y_true, y_pred = [], []
    
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        y_true.append(yb.squeeze(-1).cpu())
        y_pred.append(pred.squeeze(-1).cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    return y_true, y_pred



#main loop
def main():
    epochs = 200
    best_val = float("inf")
    best_state = None


    for epoch in range(1, epochs + 1):
        model.train()
        
        #training step
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        #validation steo
        val_mse, val_mae = evaluate(val_loader)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            tr_mse, tr_mae = evaluate(train_loader)
            print(f"Epoch {epoch:3d}; Train MSE {tr_mse:.6f}, MAE {tr_mae:.6f}; Val MSE {val_mse:.6f}, MAE {val_mae:.6f}")

    #load the best weights
    model.load_state_dict(best_state)

    #test step
    test_mse, test_mae = evaluate(test_loader)
    print(f"Test MSE {test_mse:.6f}, MAE {test_mae:.6f}")


    y_true, y_pred = prediction_collecter_plot(test_loader, model, device)

    #plotting real and predicted values of w
    plt.figure(figsize=(6,6))
    plt.scatter(y_true.numpy(), y_pred.numpy(), s=14, alpha=0.6)
    mn = min(y_true.min().item(), y_pred.min().item())
    mx = max(y_true.max().item(), y_pred.max().item())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)  # reference y=x
    plt.xlabel("True w")
    plt.ylabel("Predicted w")
    plt.title(f"Test MSE={test_mse:.6f}, MAE={test_mae:.6f}")
    plt.tight_layout()
    plt.show()


    #quite interesting plots from chat ↓
    #   # 1) Scatter: predicted vs true with y=x reference
    # plt.figure(figsize=(6,6))
    # plt.scatter(y_true, y_pred, s=10)
    # lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    # plt.plot(lims, lims, 'r--', linewidth=1, label="y = x")
    # plt.xlim(lims); plt.ylim(lims)
    # plt.xlabel("True ω"); plt.ylabel("Predicted ω̂")
    # plt.title("Test set: Predicted vs True ω")
    # plt.legend()
    # plt.show()

    # # 2) Line plot: true vs predicted, sorted by true ω (easy visual tracking)
    # order = np.argsort(y_true)
    # plt.figure(figsize=(8,4))
    # plt.plot(y_true[order], label="True ω")
    # plt.plot(y_pred[order], label="Predicted ω̂", alpha=0.85)
    # plt.xlabel("Test sample (sorted by true ω)")
    # plt.ylabel("ω")
    # plt.title("Test set: ω curves (true vs predicted)")
    # plt.legend()
    # plt.show()




main()