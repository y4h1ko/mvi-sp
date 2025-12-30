from .imports_and_libraries import *

#general utility functions
def set_device() -> torch.device:
    """Select computation device (cuda if available, otherwise CPU)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    return device

def split_and_load(dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''Splits dataset into train, validation and test parts (60%, 20%, 20%) and returns corresponding dataloaders'''

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

#functions for model without flow head
@torch.no_grad()
def evaluate(loader, model, device) -> tuple[float, float]:
    """
    Evaluate a deterministic model using MSE and MAE.

    Parameters
    ----------
    loader : DataLoader
        DataLoader providing evaluation samples.
    model : torch.nn.Module
        Trained model returning point predictions.
    device : torch.device
        Device on which evaluation is performed.

    Returns
    -------
    mse : float
        Mean squared error over the dataset.
    mae : float
        Mean absolute error over the dataset.
    """

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
def prediction_collecter_plot(loader, model, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect true and predicted target values for visualization.

    Used for plotting predicted frequencies against real-ones.

    Parameters
    ----------
    loader : DataLoader
        DataLoader providing evaluation samples.
    model : torch.nn.Module
        Trained model returning point predictions.
    device : torch.device
        Device on which inference is performed.

    Returns
    -------
    y_true : torch.Tensor
        Ground-truth target values.
    y_pred : torch.Tensor
        Corresponding model predictions.
    """

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

def train_and_eval_training(train_loader, val_loader, device, model, criterion, optimizer, scheduler, 
                            max_epochs: int=cfg.epochs, print_update: bool=False) -> tuple[nn.Module, list[float], list[float]]:
    """
    Train a deterministic model and select the best epoch based on validation MSE.

    The model is trained for a fixed number of epochs and the parameters
    corresponding to the lowest validation MSE are restored at the end.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    device : torch.device
        Device used for training.
    model : torch.nn.Module
        Model to be trained.
    criterion : callable
        Loss function (e.g. MSE).
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    max_epochs : int, optional
        Number of training epochs. Defaults to `cfg.epochs`.
    print_update : bool, optional
        If True, print progress every 10 epochs.

    Returns
    -------
    model : torch.nn.Module
        Trained model with best validation weights.
    train_mse_hist : list[float]
        Training MSE history.
    val_mse_hist : list[float]
        Validation MSE history.
    """

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


#everything for flow head with one frequency
def train_and_eval_training_flow(train_loader, val_loader, device, model, optimizer, scheduler, 
                            max_epochs: int=cfg.epochs, print_update: bool=False) -> tuple[nn.Module, list[float], list[float]]:
    """
    Train a model with a normalizing flow output head using NLL loss.

    The best model is selected based on validation MSE.

    Returns
    -------
    model : torch.nn.Module
        Trained model with best validation weights.
    train_mse_hist : list[float]
        Training MSE history.
    val_mse_hist : list[float]
        Validation MSE history.
    """
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
def evaluate2w(loader, model, device) -> tuple[float, float]:
    """
    Evaluate models predicting unordered pairs of frequencies {w1, w2}.
    The error is computed using the permutation that minimizes the loss.

    Returns
    -------
    mse : float
        Permutation-invariant mean squared error.
    mae : float
        Permutation-invariant mean absolute error.
    """

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
def prediction_collecter_plot_2w(loader, model, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect aligned ground-truth and predicted values for unordered 2D targets {w1, w2}.

    For each sample, the function chooses the permutation (w1, w2) vs (w2, w1)
    that minimizes the squared error between prediction and target, and returns
    the aligned tensors. This is useful for plotting predictions against truth.

    Parameters
    ----------
    loader : DataLoader
        DataLoader providing batches (xb, yb) where yb has shape (batch_size, 2).
    model : torch.nn.Module
        Trained model returning predictions of shape (batch_size, 2).
    device : torch.device
        Device on which inference is performed.

    Returns
    -------
    y_true : torch.Tensor
        Aligned ground-truth targets of shape (N, 2) on CPU.
    y_pred : torch.Tensor
        Aligned predictions of shape (N, 2) on CPU.
    """
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

def perm_invariant_mse(pred, target) -> torch.Tensor:
    '''Compute permutation-invariant MSE for unordered pairs {w1, w2}.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape (batch_size, 2).
    target : torch.Tensor
        Targets of shape (batch_size, 2), treated as an unordered set per row.

    Returns
    -------
    torch.Tensor
        Scalar tensor: mean permutation-invariant squared error over the batch.
    '''

    se1 = (pred - target).pow(2).sum(dim=1)
    se2 = (pred - target.flip(dims=[1])).pow(2).sum(dim=1)
    se_min = torch.minimum(se1, se2)
    return se_min.mean()

def train_and_eval_training_flow2(train_loader, val_loader, device, model, optimizer, scheduler, max_epochs: int=cfg.epochs, 
                    print_update: bool=False) -> tuple[nn.Module, list[float], list[float]]:
    """
    Train a model for unordered 2D frequency targets {w1, w2} using a flow-based NLL objective.

    The model is trained using negative log-likelihood (NLL) computed by `model.log_prob`.
    Validation is evaluated with a permutation-invariant metric (best matching of {w1, w2}).
    The best model parameters are selected based on lowest validation MSE and restored at the end.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader returning (xb, yb) with yb shape (batch_size, 2).
    val_loader : DataLoader
        Validation data loader returning (xb, yb) with yb shape (batch_size, 2).
    device : torch.device
        Device used for training and evaluation.
    model : torch.nn.Module
        Flow-based model providing `log_prob(x, y)` for NLL training.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler stepped once per epoch.
    max_epochs : int, optional
        Number of training epochs. Defaults to `cfg.epochs`.
    print_update : bool, optional
        If True, prints progress every 10 epochs.

    Returns
    -------
    model : torch.nn.Module
        Trained model with best validation weights loaded.
    train_mse_hist : list[float]
        Training MSE history (per epoch, permutation-invariant).
    val_mse_hist : list[float]
        Validation MSE history (per epoch, permutation-invariant).
    """

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

