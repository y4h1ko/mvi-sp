from .imports_and_libraries import *



def set_device():
    '''Set device to cuda if available else cpu'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device.type) 
    return device

@torch.no_grad()
def evaluate(loader, model, device):
    '''Evaluation step returning MSE and MAE'''

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
    '''Collect predictions and true values from loader just for plotting true omegas vs predicted omegas'''

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
    '''Splits dataset into train, validation and test parts'''

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

def train_and_eval_training(train_loader, val_loader, device, model, criterion, optimizer, scheduler, 
                            max_epochs: int=cfg.epochs, print_update: bool=False):
    '''Train the model and evaluate on validation. Saves best model based on validation MSE through training'''

    #real loop and training and everything....
    best_val = float("inf")
    best_state = None

    train_mse_hist, val_mse_hist = [], []

    for epoch in range(1, max_epochs + 1):
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

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if print_update:
            if epoch % 10 == 0:
                tr_mse, tr_mae = evaluate(train_loader, model, device)
                print(f"Epoch {epoch:3d}; Train MSE {tr_mse:.6f}, MAE {tr_mae:.6f}; Val MSE {val_mse:.6f}, MAE {val_mae:.6f}")

    #load the best weights
    model.load_state_dict(best_state)

    return model, train_mse_hist, val_mse_hist


#everything for flow head
def train_and_eval_training_flow(train_loader, val_loader, device, model, optimizer, scheduler, 
                            max_epochs: int=cfg.epochs, print_update: bool=False):
    '''Train the model and evaluate on validation. Saves best model based on validation MSE through training but with NLL loss function'''

    #real loop and training and everything....
    best_val = float("inf")
    best_state = None

    train_mse_hist, val_mse_hist = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        #training step
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            #NLL loss function
            log_p = model.log_prob(xb, yb)
            loss = -log_p.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        tr_mse, tr_mae = evaluate(train_loader, model, device)
        val_mse, val_mae = evaluate(val_loader, model, device)

        train_mse_hist.append(tr_mse)
        val_mse_hist.append(val_mse)

        #validation steo
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if print_update:
            if epoch % 10 == 0:
                tr_mse, tr_mae = evaluate(train_loader, model, device)
                print(f"Epoch {epoch:3d}; Train MSE {tr_mse:.6f}, MAE {tr_mae:.6f}; Val MSE {val_mse:.6f}, MAE {val_mae:.6f}")

    #load the best weights
    model.load_state_dict(best_state)

    return model, train_mse_hist, val_mse_hist


#everything for 2 omegas with flow head down there
@torch.no_grad()
def evaluate2w(loader, model, device):
    """Evaluation for 2D frequency targets that are unordered sets {w1, w2}."""

    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)

        #squared errors
        se1 = (pred - yb) ** 2
        se2 = (pred - yb.flip(dims=[1])) ** 2

        se1_sum = se1.sum(dim=1)
        se2_sum = se2.sum(dim=1)
        se_min = torch.minimum(se1_sum, se2_sum)

        #absolute errors
        ae1 = (pred - yb).abs().sum(dim=1)
        ae2 = (pred - yb.flip(dims=[1])).abs().sum(dim=1)
        ae_min = torch.minimum(ae1, ae2)

        mse_sum += se_min.sum().item()
        mae_sum += ae_min.sum().item()
        n += yb.numel()

    mse = mse_sum / n
    mae = mae_sum / n
    return mse, mae

@torch.no_grad()
def prediction_collecter_plot_2w(loader, model, device):
    '''Collect y_true, y_pred for 2D unordered targets {w1, w2}, aligned per sample using the better of the two permutations.'''
    model.eval()
    y_true_list, y_pred_list = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)

        #permutations handling
        se1 = (pred - yb) ** 2
        se2 = (pred - yb.flip(dims=[1])) ** 2
        se1_sum = se1.sum(dim=1)
        se2_sum = se2.sum(dim=1)

        use_orig = (se1_sum <= se2_sum).unsqueeze(1)

        aligned_true = torch.where(use_orig, yb, yb.flip(dims=[1]))
        aligned_pred = torch.where(use_orig, pred, pred.flip(dims=[1]))

        y_true_list.append(aligned_true.cpu())
        y_pred_list.append(aligned_pred.cpu())

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    return y_true, y_pred

def perm_invariant_mse(pred, target):
    se1 = (pred - target).pow(2).sum(dim=1)
    se2 = (pred - target.flip(dims=[1])).pow(2).sum(dim=1)
    se_min = torch.minimum(se1, se2)
    return se_min.mean()

def train_and_eval_training_flow2(train_loader, val_loader, device, model, optimizer, scheduler, max_epochs: int=cfg.epochs, print_update: bool=False):
    '''Train the model and evaluate on validation. Saves best model based on validation MSE through training but with NLL loss function'''

    #real loop and training and everything....
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_mse_hist, val_mse_hist = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        #training step
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            #symmetric NLL
            log_p = model.log_prob(xb, yb)
            loss = -log_p.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        tr_mse, tr_mae = evaluate2w(train_loader, model, device)
        val_mse, val_mae = evaluate2w(val_loader, model, device)

        train_mse_hist.append(tr_mse)
        val_mse_hist.append(val_mse)

        #validation steo
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if print_update:
            if epoch % 10 == 0:
                tr_mse, tr_mae = evaluate2w(train_loader, model, device)
                print(f"Epoch {epoch:3d}; Train MSE {tr_mse:.6f}, MAE {tr_mae:.6f}; Val MSE {val_mse:.6f}, MAE {val_mae:.6f}")

    #load the best weights
    model.load_state_dict(best_state)
    return model, train_mse_hist, val_mse_hist

