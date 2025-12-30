from ..code_files.imports_and_libraries import *
from ..code_files.dataset_creation import *
from ..code_files.positional_encodings import *
from ..code_files.models import *
from ..code_files.train_and_test import *
from ..code_files.visualizations import *

#diff_epoch = [80, 90, 100, 120, 140, 160, 180, 200]

#function to find best epoch num in run
def run_with_max_epochs(max_epochs):
    '''Training model with given max_epochs and returning best epoch, best val loss and test mse.
    Only used for analysis how much epochs to use.'''

    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel1().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

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
def epoch_sweep(epoch_settings: list[int], N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, 
                     w_max: float=cfg.omega_max, seed=cfg.seed, folder=cfg.plots_dir, save_plot: bool=False, show_plot: bool=False):
    '''Function to find optimal epoch number by iterating through different max epoch settings.
    Saves plot of max_epochs vs best val loss.'''

    results = []

    for max_ep in epoch_settings:
        best_epoch, best_val, test_mse = run_with_max_epochs(max_ep)
        print(f"max_ep={max_ep:3d} -> best_epoch={best_epoch:3d}, "
              f"best_val={best_val:.6f}, test_mse={test_mse:.6f}")
        results.append((max_ep, best_val, test_mse))

    #pltting epochs vs best val_loss
    plt.figure(figsize=(7,4))
    plt.plot([r[0] for r in results], [r[1] for r in results], marker="o")
    plt.xlabel("Max epochs")
    plt.ylabel("Best Val MSE")
    plt.ylim(bottom=0, top=0.006)
    plt.title("Finding optimal EPOCHS")
    plt.tight_layout()

    if save_plot:
        plt.savefig(folder / f"T1_EPOCHS_w{w_min}-{w_max}_N{N}_tdis{t_disc}_seed{seed}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()



def run_N_and_tdisc(N: int, t_disc: int, max_epochs: int=cfg.epochs):
    '''Function to run training and evaluation for given N and t_disc. Only used for analysis how N and t_disc affect results.'''

    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel1().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

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

def sweep_N_and_tdisc(N_values: list[int], t_disc_values: list[int], max_epochs: int=cfg.epochs):
    '''Function to iterate through different N and t_disc values to see how they affect results.'''

    all_results = []

    for N in N_values:
        for t_disc in t_disc_values:
            res = run_N_and_tdisc(N, t_disc, max_epochs=max_epochs)
            all_results.append(res)

    return all_results



def print_results_table(results):
    '''Prints a summary table of results from N and t_disc sweep. Only used with sweep_N_and_tdisc function.'''

    print("\nSummary (N, t_disc -> best_epoch, best_val, test_MSE, test_MAE):")
    for r in results:
        print( f"N={r['N']:4d}, t_disc={r['t_disc']:3d} -> best_epoch={r['best_epoch']:3d}, "
            f"best_val={r['best_val']:.6f}, test_mse={r['test_mse']:.6f}, test_mae={r['test_mae']:.6f}")

def print_model_sweep_table(results):
    '''Prints a summary table of results from model hyperparameter sweep. Only used with sweep_model_hparams function.'''

    print("\nSummary (d_model, nhead, num_layers, dim_f -> best_epoch, best_val, test_mse, test_mae)")
    
    for r in results:
        print(f"d_model={r['d_model']:3d}, nhead={r['nhead']:2d}, layers={r['num_layers']:2d}, dim_f={r['dim_f']:4d} "
            f"-> best_epoch={r['best_epoch']:3d}, best_val={r['best_val']:.6f}, test_mse={r['test_mse']:.6f}, test_mae={r['test_mae']:.6f}")
        


def run_model_config(N: int, t_disc: int, d_model: int, nhead: int, num_layers: int, dim_f: int, dropout: float=cfg.dropout, max_epochs: int=cfg.epochs):
    '''Train model with given hyperparameters and return results. Used for model hyperparameter analysis.'''

    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel1(seq_len=t_disc, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_f=dim_f, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


    #creating dataset and converting to tensor dataset
    V_np, tar_np, t_np = make_sine_dataset(N=N, t_k=t_disc)
    ds_full = from_array_to_tensor_dataset(V_np, tar_np)


    #splitting to train, val and test parts
    train_loader, val_loader, test_loader = split_and_load(ds_full)

    #training and validation steps
    model, train_mse_hist, val_mse_hist = train_and_eval_training(train_loader, val_loader, device, model, criterion, optimizer, 
                                            scheduler, print_update=False, max_epochs=max_epochs)

    #test step
    test_mse, test_mae = evaluate(test_loader, model, device)

    best_epoch = int(np.argmin(val_mse_hist) + 1)
    best_val = float(np.min(val_mse_hist))

    return {"N": N, "t_disc": t_disc, "d_model": d_model, "nhead": nhead, "num_layers": num_layers, "dim_f": dim_f, "dropout": dropout, "max_epochs": max_epochs,
        "best_epoch": best_epoch, "best_val": best_val, "test_mse": test_mse, "test_mae": test_mae, "val_curve": val_mse_hist, "train_curve": train_mse_hist, }

def sweep_model_hparams(N: int, t_disc: int, d_model_list: list[int], nhead_list: list[int], num_layers_list: list[int], 
                        dim_f_list: list[int], dropout: float=cfg.dropout, max_epochs: int=cfg.epochs):
    '''Function to iterate through different model hyperparameters (d_model, nhead, num_layers, dim_f) and collect results.
    It will do all combinations of given hyperparameter lists → bruteforce search.
    
    Results are also saved to CSV file defined in cfg.csv_path.
    Saves loss curves plots for each configuration with different zoom levels. Recomended to run through night'''
    
    all_results = []

    #iterate all combinations
    for d_model in d_model_list:
        for nhead in nhead_list:
            for num_layers in num_layers_list:
                for dim_f in dim_f_list:
                    
                    res = run_model_config(N=N, t_disc=t_disc, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_f=dim_f, dropout=dropout, max_epochs=max_epochs, )
                    all_results.append(res)
                    
                    
                    file_exists = cfg.csv_path.exists()
                    with open(cfg.csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["N","t_disc","d_model","nhead","num_layers","dim_f","dropout","max_epochs","best_epoch","best_val","test_mse","test_mae"])
                        writer.writerow([N, t_disc, d_model, nhead, num_layers, dim_f, dropout, max_epochs, res["best_epoch"], res["best_val"], res["test_mse"], res["test_mae"]])


                    print(f"\nN={N}, t_disc={t_disc}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_f={dim_f}, dropout={dropout}, max_epochs={max_epochs}")
                    print(f"best_epoch={res['best_epoch']}, best_val={res['best_val']:.6f}, test_mse={res['test_mse']:.6f}, test_mae={res['test_mae']:.6f}")

                    plot_loss_curves(res["train_curve"], res["val_curve"], save_plot=True, epochs=len(res["val_curve"]), N=res["N"], t_disc=res["t_disc"],
                        name_suf=f"_dmodel{res['d_model']}_nhead{res['nhead']}_layers{res['num_layers']}_dimf{res['dim_f']}", zoom="full")
                    plot_loss_curves(res["train_curve"], res["val_curve"], save_plot=True, epochs=len(res["val_curve"]), N=res["N"], t_disc=res["t_disc"],
                        name_suf=f"_dmodel{res['d_model']}_nhead{res['nhead']}_layers{res['num_layers']}_dimf{res['dim_f']}", y_limit=0.1, zoom=f"0.1")
                    plot_loss_curves(res["train_curve"], res["val_curve"], save_plot=True, epochs=len(res["val_curve"]), N=res["N"], t_disc=res["t_disc"],
                        name_suf=f"_dmodel{res['d_model']}_nhead{res['nhead']}_layers{res['num_layers']}_dimf{res['dim_f']}", y_limit=0.01, zoom="0.01")

    return all_results

def tell_me_model_params(model=TransformerModel1()):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in {model}: {total_params}")

def tell_model_size(model=TransformerModel1()):
    
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Size: {:.3f} MB'.format(size_all_mb))


def main1(epoch_list: list[int]):
    '''Run epoch sweep analysis with given epoch list.'''

    epoch_sweep(epoch_list, save_plot=True)

def main2(N_list: list[int], t_disc_list: list[int]):
    '''Run N and t_disc sweep analysis with given N and t_disc lists.'''

    results = sweep_N_and_tdisc(N_list, t_disc_list, max_epochs=cfg.epochs)
    print_results_table(results)

    # plots
    for N_val in N_list:
        plot_val_curves_fixed_N(results, N=N_val, save_plot=True)
        plot_val_curves_fixed_N(results, N=N_val, save_plot=True, y_limit=0.05, zoom=f"0.05")
        plot_val_curves_fixed_N(results, N=N_val, save_plot=True, y_limit=0.01, zoom=f"0.01")

def main3():
    '''Run model hyperparameter sweep analysis with predefined hyperparameter lists. When run, it will take a long time as 
    it does bruteforce search through all combinations. Needs to configure hyperparameter lists in the function.'''

    #smaller configs for faster testing
    N = cfg.num_of_samples
    t_disc = cfg.discr_of_time
    max_epochs = cfg.epochs
    dropout = cfg.dropout

    #different hyperparameters
    d_model_list = [32, 64, 128, 256]
    nhead_list = [1, 2, 4, 8]
    num_layers_list = [1, 2, 3, 4]
    dim_f_list = [64, 128, 256, 512]

    results = sweep_model_hparams( N=N, t_disc=t_disc, d_model_list=d_model_list, nhead_list=nhead_list, num_layers_list=num_layers_list, 
                                  dim_f_list=dim_f_list, dropout=dropout, max_epochs=max_epochs,)

    
    print_model_sweep_table(results)

    best = min(results, key=lambda r: r["best_val"])
    epochs_axis = range(1, len(best["val_curve"]) + 1)

    plot_loss_curves(best["train_curve"], best["val_curve"], save_plot=True, epochs=len(best["val_curve"]), N=best["N"], t_disc=best["t_disc"],
                     name_suf=f"_BESTHPAR_dmodel{best['d_model']}_nhead{best['nhead']}_layers{best['num_layers']}_dimf{best['dim_f']}")

def main4(save: bool=False, show: bool=False):
    '''Plot example sine wave with and without noise for frequency w=1.'''

    V_clean, _ , t = make_sine_dataset(N=1, w_min=1, w_max=1, noise=False)
    V_noisy, _, _ = make_sine_dataset(N=1, w_min=1, w_max=1, noise=True)

    plot_wave_samples(t=t, V_clean=V_clean[0], V_noisy=V_noisy[0], save_plot=save, show_plot=show)

def main5():
    '''Plot multi axis parallel plot - used for hyperparam ploting'''
    
    plot_parallel_hparams(csv_path="plots/rep04-different-model-hyperparameters/T1_hyperprams.csv", renderer='browser', show=False, 
                      save_path="plots/rep05-something-small/parallel_all.html")

    plot_parallel_hparams(csv_path="plots/rep04-different-model-hyperparameters/T1_hyperprams.csv", top_k=8, renderer='browser', show=False, 
                      save_path="plots/rep05-something-small/parallel_top8_zoomed.html")


def main6():
    '''used to tune flow hyperparameters... but '''
    dmodel_list = [64]
    nhead_list = [2]
    num_of_layers_list = [2]
    dim_forward_list = [64]
    flow_hid_features_list = [128]
    flow_layers_list = [4]
    results = []

    for a in dmodel_list:
        for b in nhead_list:
            for c in num_of_layers_list:
                for d in dim_forward_list:
                    for e in flow_hid_features_list:
                        for f in flow_layers_list:
                            set_seed()
                            device = set_device()
                            #V_np, tar_np, t_np = make_sine_dataset(noise=True)
                            V_np, tar_np, t_np = make_double_sine_dataset(noise=True)
                            ds_full = from_array_to_tensor_dataset(V_np, tar_np)
                            train_loader, val_loader, test_loader = split_and_load(ds_full)

                            #model = TransformerModel2(d_model=a, nhead=b, num_layers=c, dim_f=d, flow_hidden_features=e, flow_num_layers=f).to(device)
                            model = TransformerModel3(d_model=a, nhead=b, num_layers=c, dim_f=d, flow_hidden_features=e, flow_num_layers=f).to(device)
                            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

                            #model, train_mse_hist, val_mse_hist = train_and_eval_training_flow(train_loader, val_loader, device, model, optimizer, scheduler)
                            model, train_mse_hist, val_mse_hist = train_and_eval_training_flow2(train_loader, val_loader, device, model, optimizer, scheduler, lambda_reg=0.0)

                            if min(val_mse_hist) == "nan":
                                continue

                            test_mse, test_mae = evaluate2w(test_loader, model, device)
                            
                            print(f'Hyperparams: d_model: {a}, nhead: {b}, num_layers: {c}, dim_f: {d},' + 
                                  f'low_hidden: {e}, flow_layers: {f}, best_val_mse: {min(val_mse_hist)}, test_mse: {test_mse}, test_mae: {test_mae}')

                            results.append({"d_model": a, "nhead": b, "num_layers": c, "dim_f": d,
                                "flow_hidden": e, "flow_layers": f, "best_val_mse": min(val_mse_hist),
                                "test_mse": float(test_mse), "test_mae": float(test_mae) })

    df = pd.DataFrame(results)
    df.to_csv(cfg.csv_path, index=False)
    print(f"\nSaved grid search results to {cfg.csv_path}")

def main7():
    N_samples_list = [1000, 10000]
    t_disc_list = [100, 300, 500]

    a, b, c, d, e, f = 64, 2, 2, 64, 128, 4

    results = []

    for N in N_samples_list:
        for t_disc in t_disc_list:

            set_seed()
            device = set_device()
            V_np, tar_np, t_np = make_sine_dataset(N=N, t_k=t_disc, noise=True)
            ds_full = from_array_to_tensor_dataset(V_np, tar_np)
            train_loader, val_loader, test_loader = split_and_load(ds_full)

            model = TransformerModel2(seq_len=t_disc, d_model=a, nhead=b, num_layers=c, dim_f=d, flow_hidden_features=e, flow_num_layers=f).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

            model, train_mse_hist, val_mse_hist = train_and_eval_training_flow(train_loader, val_loader, device, model, optimizer, scheduler)

            test_mse, test_mae = evaluate(test_loader, model, device)

            y_true, y_pred = prediction_collecter_plot(test_loader, model, device)
            
            print(f'Hyperparams: num_of_samples: {N}, t_disc: {t_disc}, d_model: {a}, nhead: {b}, num_layers: {c}, dim_f: {d},' + 
                    f'low_hidden: {e}, flow_layers: {f}, best_val_mse: {float(min(val_mse_hist))}, test_mse: {test_mse}, test_mae: {test_mae}')

            results.append({"d_model": a, "nhead": b, "num_layers": c, "dim_f": d,
                        "flow_hidden": e, "flow_layers": f, "num_of_samples": N, "t_disc": t_disc, 
                        "best_val_mse": float(min(val_mse_hist)), "test_mse": float(test_mse), "test_mae": float(test_mae) })
            
            #plots - save or show option
            plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, N=N, t_disc=t_disc, save_plot=True, show_plot=False)
            plot_loss_curves(train_mse_hist, val_mse_hist, N=N, t_disc=t_disc, save_plot=True, show_plot=False)
            plot_loss_curves(train_mse_hist, val_mse_hist, N=N, t_disc=t_disc, save_plot=True, show_plot=False, y_limit=0.025, zoom='0.025')

            plot_dataset_vs_learned_marginal(model, device, test_loader, N=N, t_disc=t_disc, save_plot=True, show_plot=False)

            for idx in range(0, int(0.2*N), int(0.0125*N)):
                plot_flow_posterior_one_example(model, device, test_loader, global_index=idx, N=N, t_disc=t_disc, save_plot=True, show_plot=False)
            
            plot_error_vs_true_omega(y_true, y_pred, N=N, t_disc=t_disc, save_plot=True, show_plot=False)


    df = pd.DataFrame(results)
    df.to_csv(cfg.csv_path, index=False)
    print(f"\nSaved grid search results to {cfg.csv_path}")

#with this i tried to find optimal epoch number
#diff_epoch = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400]
#main1(diff_epoch)


#with this i tried to find how N and t_disc affect loss function
#N_values = [1000, 2500, 5000, 7500, 10000]
#t_disc_values = [60, 70, 85, 100, 125, 150]
#main2(N_values, t_disc_values)


#with this i tried to find best model hyperparameters for N=1000 and t_disc=100
#diff hyperparams in the main function
#main3()

#just to show wave
#main4(show=True)

#how many trainable parameters has the model...
#tell_me_model_params(model=TransformerModel1(d_model=256, nhead=8, num_layers=1, dim_f=256))
#tell_me_model_params(model=TransformerModel1(d_model=64, nhead=2, num_layers=2, dim_f=64))

#to plot hyperparameters at once
#main5()

#analyze_flow_distributions(TransformerModel2(), num_flow_samples=1000, noise=True, show_plots=True)

#searching across hyperparameters in T3
#main6()

#fine-tunned T2 with different N and t_disc → searching pattern in bigger/smaller omegas
#main7()


#plotting different example waves with clean and noisy points
#plot_waves_clean_and_signal_points(i=18, wave_type="product", noise=True, save_plot=False, show_plot=True)
#plot_waves_clean_and_signal_points(i=5, wave_type="linear", noise=True, save_plot=False, show_plot=True)
#plot_waves_clean_and_signal_points(i=4, wave_type="single", noise=True, save_plot=False, show_plot=True)
plot_analytic_contours_sin2(t0=2.0, signal="linear", triple_t=True, dt=1.0, levels=10, fullcolor=False, save_plot=True, show_plot=False)
plot_analytic_contours_sin2(t0=2.0, signal="product", triple_t=True, dt=1.0, levels=10, fullcolor=False, save_plot=True, show_plot=False)
#plot_analytic_contours_sin2(t0=2.0, signal="linear", triple_t=True, dt=1.0, save_plot=True, show_plot=False)
#plot_analytic_contours_sin2(t0=2.0, signal="product", triple_t=True, dt=1.0, save_plot=True, show_plot=False)

