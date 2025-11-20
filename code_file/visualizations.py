from .imports_and_libraries import *
from .dataset_creation import *


def plot_wave_samples(t, V_clean, V_noisy, w: float=1.0, mu: float=cfg.mu, sigma: float=cfg.noise_std, save_plot: bool=False, show_plot: bool=False):
    '''Plot a single sine wave with and without Gaussian noise as example.
    
    - save_plot: when True, saves the plot to the specified folder as .png
    - show_plot: when True, displays the plot on the screen
    
    Saved plot name is like: 'sine_with_noise_mu{mu}_sigma{sigma}_example.png'''

    plt.figure(figsize=(8,6))
    plt.plot(t, V_clean, label="clean sine", linewidth=2)
    plt.scatter(t, V_noisy, s=15, alpha=0.7, label="noisy sample", c='black', marker='x')

    plt.xlabel("t")
    plt.ylabel("sin(w*t)")
    plt.title(f"Sine example with noise (w={w:.3f}, mu={mu}, sigma={sigma})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(cfg.plots_dir / f"sine_with_noise_mu{mu}_sigma{sigma}_example.png", dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, 
                      w_max: float=cfg.omega_max, seed=cfg.seed, sigma: float=cfg.noise_std, folder=cfg.plots_dir, save_plot: bool=False, show_plot: bool=False):
    '''Plot predicted vs true values frequencies into scatter plot from test set.
    
    - save_plot: if True, saves the plot to the specified folder as .png
    - show_plot: if True, displays the plot on the screen
    - other optional parameters are for plot title and filename

    Saved plot name is like: 'T1_w{w_min}-{w_max}_N{N}_tdis{t_disc}_seed{seed}_PREDvsREAL.png'
    '''

    plt.figure(figsize=(6,6))
    plt.scatter(y_true.numpy(), y_pred.numpy(), s=14, alpha=0.6)
    mn = min(y_true.min().item(), y_pred.min().item())
    mx = max(y_true.max().item(), y_pred.max().item())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.grid(True, which="both")
    plt.xlabel("True w")
    plt.ylabel("Predicted w")
    plt.title(f"Test N={N}, w=[{w_min}-{w_max}], tdis={t_disc}\nMSE={test_mse:.6f}, MAE={test_mae:.6f}, std={sigma}")
    plt.tight_layout()
    if save_plot:
        plt.savefig(folder / f"T1_w{w_min}-{w_max}_N{N}_tdis{t_disc}_seed{seed}_std{sigma}_PREDvsREAL.png", dpi=300)
    if show_plot:
        plt.show()
    
    plt.close()


def plot_loss_curves(train_mse_hist, val_mse_hist, epochs: int=cfg.epochs, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, 
                     w_max: float=cfg.omega_max, seed=cfg.seed, folder=cfg.plots_dir, sigma: float=cfg.noise_std, save_plot: bool=False, show_plot: bool=False, 
                     y_limit: float=None, zoom: str="full", name_suf: str=""):
    '''Plot training and validation loss curves over epochs.
    
    - save_plot: if True, saves the plot to the specified folder as .png
    - show_plot: if True, displays the plot on the screen
    - y_limit: sets the y-axis limit for better detail
    - zoom: defualt is 'full', when y_limit is set, recomended to set as y_limit value for recognition in filename
    - other optional parameters are for plot title and filename
    
    Saved plot name is like: 'T1{name_suf}_w{w_min}-{w_max}_N{N}_tdis{t_disc}_seed{seed}_LOSSf_{zoom}.png'
    '''


    epochs_axis = range(1, epochs + 1)
    plt.figure(figsize=(8,5))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.plot(epochs_axis, train_mse_hist, label="Train MSE")
    plt.plot(epochs_axis, val_mse_hist,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.ylim(bottom=0, top=y_limit)
    plt.title(f"Training/Validation Loss \n N={N}, w=[{w_min}-{w_max}], tdis={t_disc}, std{sigma}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(folder / f"T1{name_suf}_std{sigma}_w{w_min}-{w_max}_N{N}_tdis{t_disc}_seed{seed}_LOSSf_{zoom}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()


def plot_val_curves_fixed_N(results, N, folder=cfg.plots_dir, save_plot: bool=False, show_plot: bool=False, y_limit: float=None, zoom: str="full"):
    '''Plot Val MSE vs epoch for all t_disc, for a given N. Only used when analyzing different number of samples N and time discretizations t_disc.
    
    - save_plot: if True, saves the plot to the specified folder as .png
    - show_plot: if True, displays the plot on the screen
    - y_limit: sets the y-axis limit for better detail
    - zoom: defualt is 'full', when y_limit is set, recomended to set as y_limit value for recognition in filename

    Saved plot name is like: 'VALcurves_N{N}_{zoom}.png'
    '''

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
        plt.savefig(folder / f"VALcurves_N{N}_{zoom}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()


def plot_parallel_hparams( csv_path: str, top_k: int | None = None, renderer: str = "browser",
    dims: list[str] | None = None, color_col: str = "best_val", title_prefix: str = "Parallel coordinates", show: bool = False, save_path: str | None = None):
    """
    Plot parallel coordinates for transformer hyperparameter search.

    Parameters
    ----------
    csv_path : str -Ppth to CSV file with reuslts.
    top_k : int or None -None use ALL rows. -int use n rows with smallest best_val.
    rescale_to_subset : bool
        If True  -> colour scale and axis range use only the subset (top_k).
        If False -> colour scale and axis range use the FULL dataset.
    
    dims : list[str] or None
        List of columns to use as axes. If None, defaults to:
        ['d_model', 'nhead', 'num_layers', 'dim_f', color_col]
        
    title_prefix : str -Text at beginning of figure title.
    """

    pio.renderers.default = renderer

    df = pd.read_csv(csv_path)

    if dims is None:
        dims = ["d_model", "nhead", "num_layers", "dim_f", color_col]

    # choose subset
    if top_k is None:
        subset = df
        suffix = "ALL models"
    else:
        subset = df.nsmallest(top_k, color_col)
        suffix = f"TOP {top_k} models (lowest {color_col})"

    # global range for colour/axis
    global_min = df[color_col].min()
    global_max = df[color_col].max()

    cmin = subset[color_col].min()
    cmax = subset[color_col].max()

    fig = px.parallel_coordinates(subset[dims], dimensions=dims, color=color_col, color_continuous_scale="Viridis", range_color=(cmax, cmin))

    fig.update_layout(title=f"{title_prefix} – {suffix}", width=1800, height=900)

    if save_path is not None:
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, width=1800, height=900, scale=2.0)

    if show:
        fig.show()