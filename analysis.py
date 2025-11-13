from code_file.imports_and_libraries import *
from code_file.dataset_creation import *
from code_file.positional_encodings import *
from code_file.models import *
from code_file.train_and_test import *
from code_file.visualizations import *

#diff_epoch = [80, 90, 100, 120, 140, 160, 180, 200]

#function to find best epoch num in run
def run_with_max_epochs(max_epochs):
    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel1(seq_len=DISCR_OF_TIME, d_model=128, nhead=4, num_layers=2, dim_f=256, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    #creating dataset and converting to tensor dataset
    V_np, tar_np, t_np = make_sine_dataset()
    ds_full = from_array_to_tensor_dataset(V_np, tar_np)

    #splitting to train, val and test parts
    train_loader, val_loader, test_loader = split_and_load(ds_full)

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
    plt.ylabel("Best Val MSE")
    plt.ylim(bottom=0, top=0.006)
    plt.title("Finding optimal EPOCHS")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"T1_EPOCHS_w{OMEGA_MIN}-{OMEGA_MAX}_N{NUM_OF_SAMPLES}_tdis{DISCR_OF_TIME}_seed{SEED}.png", dpi=300)


def run_N_and_tdisc(N: int, t_disc: int, max_epochs: int = EPOCHS):
    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel1(seq_len=t_disc, d_model=128, nhead=4, num_layers=2, dim_f=256, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    #creating dataset and converting to tensor dataset
    V_np, tar_np, t_np = make_sine_dataset(N=N, t_k=t_disc)
    ds_full = from_array_to_tensor_dataset(V_np, tar_np)

    #splitting to train, val and test parts
    train_loader, val_loader, test_loader = split_and_load(ds_full)

    #training and validation steps
    model, train_mse_hist, val_mse_hist = train_and_eval_training(train_loader, val_loader, device, model, criterion, 
                                            optimizer, scheduler, max_epochs=max_epochs, print_update=False)

    #test step
    test_mse, test_mae = evaluate(test_loader, model, device)

    #good epoch save
    best_epoch = int(np.argmin(val_mse_hist) + 1)
    best_val = float(np.min(val_mse_hist))

    return {"N": N, "t_disc": t_disc, "max_epochs": max_epochs, "best_epoch": best_epoch, "best_val": best_val, 
        "test_mse": test_mse, "test_mae": test_mae, "val_curve": val_mse_hist, "train_curve": train_mse_hist}


def sweep_N_and_tdisc(N_values: list[int], t_disc_values: list[int], max_epochs: int = EPOCHS):

    all_results = []

    for N in N_values:
        for t_disc in t_disc_values:
            res = run_N_and_tdisc(N, t_disc, max_epochs=max_epochs)
            all_results.append(res)

    return all_results

def print_results_table(results):
    # sort by N then t_disc
    #results_sorted = sorted(results, key=lambda r: (r["N"], r["t_disc"]))

    print("\nSummary (N, t_disc -> best_epoch, best_val, test_MSE, test_MAE):")
    for r in results:
        print(
            f"N={r['N']:4d}, t_disc={r['t_disc']:3d} -> "
            f"best_epoch={r['best_epoch']:3d}, "
            f"best_val={r['best_val']:.6f}, "
            f"test_mse={r['test_mse']:.6f}, "
            f"test_mae={r['test_mae']:.6f}"
        )


def main1(epoch_list: list[int]):
    epoch_sweep(epoch_list)

def main2(N_list: list[int], t_disc_list: list[int]):
    MAX_EPOCHS_FOR_SEARCH = 160

    results = sweep_N_and_tdisc(N_list, t_disc_list, max_epochs=MAX_EPOCHS_FOR_SEARCH)
    print_results_table(results)

    # plots
    for N_val in N_list:
        plot_val_curves_fixed_N(results, N=N_val, save_plot=True)
        plot_val_curves_fixed_N(results, N=N_val, save_plot=True, y_limit=0.05, zoom=f"0.05")
        plot_val_curves_fixed_N(results, N=N_val, save_plot=True, y_limit=0.01, zoom=f"0.01")


#with this i tried to find optimal epoch number
#diff_epoch = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400]
#main1(diff_epoch)


#with this i tried to find how N and t_disc affect loss function
#N_values = [1000, 2500, 5000, 7500, 10000]
#t_disc_values = [60, 70, 85, 100, 125, 150]
#main2(N_values, t_disc_values)









