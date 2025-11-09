from code_file.imports_and_libraries import *
from code_file.dataset_creation import *
from code_file.positional_encodings import *
from code_file.models import *
from code_file.train_and_test import *
from code_file.visualizations import *


#function to find best epoch num in run
def run_with_max_epochs(max_epochs):
    set_seed()
    device = set_device()

    V_np, tar_np, t_np = make_sine_dataset()
    X = torch.from_numpy(V_np).float()
    y = torch.from_numpy(tar_np).float()
    ds_full = TensorDataset(X, y)

    ds_train, ds_val, ds_test = random_split(
        ds_full, [0.6, 0.2, 0.2],
        generator=torch.Generator().manual_seed(0)
    )

    batch = min(32, max(1, len(ds_train)))
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=False)

    model = TransformerModel1(seq_len=DISCR_OF_TIME, d_model=128, nhead=4,
                              num_layers=2, dim_f=256, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val = float("inf")
    best_state = None
    val_mse_hist = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_mse, _ = evaluate(val_loader, model, device)
        val_mse_hist.append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    #load the best weights
    model.load_state_dict(best_state)
    test_mse, _ = evaluate(test_loader, model, device)

    best_epoch = int(np.argmin(val_mse_hist) + 1)
    return best_epoch, best_val, test_mse

#changing num of epochs and plot
def epoch_sweep(epoch_settings: list[int]):
    results = []

    for max_ep in epoch_settings:
        best_epoch, best_val, test_mse = run_with_max_epochs(max_ep)
        print(f"max_ep={max_ep:3d} -> best_epoch={best_epoch:3d}, "
              f"best_val={best_val:.6f}, test_mse={test_mse:.6f}")
        results.append((max_ep, best_val, test_mse))

    #pltting epochs vs best val_loss
    plt.figure(figsize=(7,4))
    plt.plot([r[0] for r in results],
             [r[1] for r in results],
             marker="o")
    plt.xlabel("Max epochs")
    plt.ylabel("Best Val MSE (within run)")
    plt.title("Sweet spot search for number of epochs")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"T1_EPOCHS_w{OMEGA_MIN}-{OMEGA_MAX}_N{NUM_OF_SAMPLES}_tdis{DISCR_OF_TIME}_seed{SEED}.png", dpi=300)


#diff_epoch = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400]

#epoch_sweep(diff_epoch)