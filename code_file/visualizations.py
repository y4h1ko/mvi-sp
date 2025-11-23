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


@torch.no_grad()
def plot_dataset_vs_learned_marginal(model: nn.Module, device, loader, num_samples_per_x: int=10, bins: int=40, save_plot: bool=False, show_plot: bool=False):
    """
    Histogram of:
      - dataset targets ω (all y from loader)
      - model samples ω ~ p(ω | x) from the flow head

    This uses model.sample(...) → **learned ω distribution**, NOT latent z.
    """
    model.eval()

    all_targets = []
    all_model_samples = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        # true ω
        all_targets.append(yb.squeeze(-1).cpu().numpy())   # [B]

        # learned distribution p(ω | x): sample ω from flow head
        samples = model.sample(xb, num_samples=num_samples_per_x)  # [B, S, 1]
        samples = samples.squeeze(-1).cpu().numpy().reshape(-1)    # flatten to [B*S]
        all_model_samples.append(samples)

    targets = np.concatenate(all_targets)
    flow_samples = np.concatenate(all_model_samples)

    plt.figure(figsize=(8, 5))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.hist(
        targets,
        bins=bins,
        density=True,
        alpha=0.7,
        label="dataset ω (targets)",
    )

    plt.hist(
        flow_samples,
        bins=bins,
        density=True,
        alpha=0.7,
        label="flow samples ω (model)",
    )

    plt.xlabel("ω")
    plt.ylabel("density")
    plt.title("Dataset vs learned ω distribution (flow head)")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        path = cfg.plots_dir / "dataset_vs_learned_marginal_flow.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


@torch.no_grad()
def plot_flow_posterior_one_example(model: nn.Module, device, loader, index_in_batch: int=0, num_samples: int=500, bins: int=40, save_plot: bool=False, show_plot: bool=False):
    """
    Take one x from the first batch, sample ω ~ p(ω | x) many times,
    and plot the learned 1D conditional with the true ω marked.
    """
    model.eval()

    xb, yb = next(iter(loader))
    xb = xb.to(device)
    yb = yb.to(device)

    x_one = xb[index_in_batch : index_in_batch + 1]
    w_true = yb[index_in_batch].item()

    samples = model.sample(x_one, num_samples=num_samples)
    samples = samples.squeeze().cpu().numpy()

    plt.figure(figsize=(7, 4))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.hist(samples, bins=bins, density=True, alpha=0.8, label="flow samples ω | x")
    plt.axvline(w_true, linestyle="--", linewidth=2, label=f"true ω = {w_true:.3f}")

    plt.xlabel("ω")
    plt.ylabel("density")
    plt.title("Learned conditional p(ω | x) for one example")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        path = cfg.plots_dir / "flow_posterior_one_example.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


@torch.no_grad()
def plot_uncertainty_vs_error( model: nn.Module, device, loader, num_samples: int=200, save_plot: bool=False, show_plot: bool=False,):
    """
    For each example:
      - draw many ω samples from p(ω | x)
      - compute predictive mean and std
      - compute |mean - true ω|
    Then scatter: std (x-axis) vs error (y-axis).
    """
    model.eval()

    all_std = []
    all_err = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        samples = model.sample(xb, num_samples=num_samples)

        if samples.dim() == 3 and samples.shape[0] == num_samples:
            samples = samples.permute(1, 0, 2)

        samples = samples.squeeze(-1)

        mean_w = samples.mean(dim=1)
        std_w  = samples.std(dim=1)
        true_w = yb.squeeze(-1)

        err = (mean_w - true_w).abs()

        all_std.append(std_w.cpu().numpy())
        all_err.append(err.cpu().numpy())

    all_std = np.concatenate(all_std)
    all_err = np.concatenate(all_err)

    plt.figure(figsize=(6, 6))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.scatter(all_std, all_err, s=12, alpha=0.6)
    plt.xlabel("predictive std(w)")
    plt.ylabel("abs(mean(w) - true w)")
    plt.title("Uncertainty vs absolute error")
    plt.tight_layout()

    if save_plot:
        path = cfg.plots_dir / "uncertainty_vs_error_flow.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()