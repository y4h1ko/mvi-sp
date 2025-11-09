from .imports_and_libraries import *


def plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, save_plot: bool=False, show_plot: bool=False):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true.numpy(), y_pred.numpy(), s=14, alpha=0.6)
    mn = min(y_true.min().item(), y_pred.min().item())
    mx = max(y_true.max().item(), y_pred.max().item())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.xlabel("True w")
    plt.ylabel("Predicted w")
    plt.title(f"Test N={NUM_OF_SAMPLES}, w=[{OMEGA_MIN}-{OMEGA_MAX}], tdis={DISCR_OF_TIME}\nMSE={test_mse:.6f}, MAE={test_mae:.6f}")
    plt.tight_layout()
    if save_plot:
        plt.savefig(PLOTS_DIR / f"T1_w{OMEGA_MIN}-{OMEGA_MAX}_N{NUM_OF_SAMPLES}_tdis{DISCR_OF_TIME}_seed{SEED}_PREDvsREAL.png", dpi=300)
    if show_plot:
        plt.show()

def plot_loss_curves(train_mse_hist, val_mse_hist, save_plot: bool=False, show_plot: bool=False):
    epochs_axis = range(1, EPOCHS + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_axis, train_mse_hist, label="Train MSE")
    plt.plot(epochs_axis, val_mse_hist,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Training/Validation Loss \n N={NUM_OF_SAMPLES}, w=[{OMEGA_MIN}-{OMEGA_MAX}], tdis={DISCR_OF_TIME}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(PLOTS_DIR / f"T1_w{OMEGA_MIN}-{OMEGA_MAX}_N{NUM_OF_SAMPLES}_tdis{DISCR_OF_TIME}_seed{SEED}_LOSSf.png", dpi=300)
    if show_plot:
        plt.show()





