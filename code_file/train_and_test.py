from .imports_and_libraries import *



def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type) 
    return device

@torch.no_grad()
def evaluate(loader, model, device):
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

def split_and_load(dataset):
    #spltting to train, val and test parts
    ds_train, ds_val, ds_test = random_split(
        dataset, [0.6, 0.2, 0.2],
        generator=torch.Generator().manual_seed(0)
    )

    batch = min(32, max(1, len(ds_train)))
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=False)

    return train_loader, val_loader, test_loader

def train_and_eval_training(train_loader, val_loader, device, model, criterion, optimizer, scheduler):
    #real loop and training and everything....
    best_val = float("inf")
    best_state = None

    train_mse_hist, val_mse_hist = [], []

    for epoch in range(1, EPOCHS + 1):
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

        tr_mse, tr_mae = evaluate(train_loader, model, device)
        val_mse, val_mae = evaluate(val_loader, model, device)

        train_mse_hist.append(tr_mse)
        val_mse_hist.append(val_mse)

        #validation steo
        val_mse, val_mae = evaluate(val_loader, model, device)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            tr_mse, tr_mae = evaluate(train_loader, model, device)
            print(f"Epoch {epoch:3d}; Train MSE {tr_mse:.6f}, MAE {tr_mae:.6f}; Val MSE {val_mse:.6f}, MAE {val_mae:.6f}")

    #load the best weights
    model.load_state_dict(best_state)

    return model, train_mse_hist, val_mse_hist