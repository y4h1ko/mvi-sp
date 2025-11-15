from .imports_and_libraries import *


def plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, save_plot: bool=False, show_plot: bool=False):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true.numpy(), y_pred.numpy(), s=14, alpha=0.6)
    mn = min(y_true.min().item(), y_pred.min().item())
    mx = max(y_true.max().item(), y_pred.max().item())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.grid(True, which="both")
    plt.xlabel("True w")
    plt.ylabel("Predicted w")
    plt.title(f"Test N={NUM_OF_SAMPLES}, w=[{OMEGA_MIN}-{OMEGA_MAX}], tdis={DISCR_OF_TIME}\nMSE={test_mse:.6f}, MAE={test_mae:.6f}")
    plt.tight_layout()
    if save_plot:
        plt.savefig(PLOTS_DIR / f"T1_w{OMEGA_MIN}-{OMEGA_MAX}_N{NUM_OF_SAMPLES}_tdis{DISCR_OF_TIME}_seed{SEED}_PREDvsREAL.png", dpi=300)
    if show_plot:
        plt.show()

def plot_loss_curves(train_mse_hist, val_mse_hist, save_plot: bool=False, show_plot: bool=False, epochs: int=EPOCHS, N: int=NUM_OF_SAMPLES, 
                     t_disc: int=DISCR_OF_TIME, name_suf: str="", y_limit: float=None, zoom: str="full"):
    epochs_axis = range(1, epochs + 1)
    plt.figure(figsize=(8,5))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.plot(epochs_axis, train_mse_hist, label="Train MSE")
    plt.plot(epochs_axis, val_mse_hist,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.ylim(bottom=0, top=y_limit)
    plt.title(f"Training/Validation Loss \n N={N}, w=[{OMEGA_MIN}-{OMEGA_MAX}], tdis={t_disc}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(PLOTS_DIR / f"T1{name_suf}_w{OMEGA_MIN}-{OMEGA_MAX}_N{N}_tdis{t_disc}_seed{SEED}_LOSSf_{zoom}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()

def plot_val_curves_fixed_N(results, N, save_plot: bool=False, show_plot: bool=False, y_limit: float=None, zoom: str="full"):
    """Plot Val MSE vs epoch for all t_disc, for a given N."""
    subset = [r for r in results if r["N"] == N]
    if not subset:
        print(f"No results for N={N}")
        return

    plt.figure(figsize=(8, 5))
    plt.grid(True, which="both")

    for r in subset:
        epochs_axis = range(1, len(r["val_curve"]) + 1)
        plt.plot(epochs_axis, r["val_curve"], label=f"t_disc={r['t_disc']}")

    plt.xlabel("Epoch")
    plt.ylabel("Val MSE")
    plt.ylim(bottom=0, top=y_limit)
    plt.title(f"Val MSE vs Epoch for N={N}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(PLOTS_DIR / f"VALcurves_N{N}_{zoom}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()



