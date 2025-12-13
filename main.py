from code_file.imports_and_libraries import *
from code_file.dataset_creation import *
from code_file.positional_encodings import *
from code_file.models import *
from code_file.train_and_test import *
from code_file.visualizations import *

#main loop
def main1(plot1: bool = False, plot2: bool = False):
    #setup for reproducibility and device
    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel1().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    #creating dataset and converting to tensor dataset
    V_np, tar_np, t_np = make_sine_dataset(noise=True)
    ds_full = from_array_to_tensor_dataset(V_np, tar_np)


    #splitting to train, val and test parts
    train_loader, val_loader, test_loader = split_and_load(ds_full)

    #training and validation steps
    model, train_mse_hist, val_mse_hist = train_and_eval_training(train_loader, val_loader, device, model, criterion, optimizer, scheduler)

    #test step
    test_mse, test_mae = evaluate(test_loader, model, device)
    print(f"Test MSE {test_mse:.6f}, MAE {test_mae:.6f}")

    #plotting data collection
    y_true, y_pred = prediction_collecter_plot(test_loader, model, device)

    #plots - save or show option
    if plot1:
        plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, save_plot=True, show_plot=False)
    if plot2:
        plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=True)
        plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=True, y_limit=0.025)



#main loop
def main2():
    #setup for reproducibility and device
    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    #creating dataset and converting to tensor dataset
    V_np, tar_np, t_np = make_sine_dataset(noise=True)
    ds_full = from_array_to_tensor_dataset(V_np, tar_np)

    #splitting to train, val and test parts
    train_loader, val_loader, test_loader = split_and_load(ds_full)

    #training and validation steps
    model, train_mse_hist, val_mse_hist = train_and_eval_training_flow(train_loader, val_loader, device, model, optimizer, scheduler)

    #test step
    test_mse, test_mae = evaluate(test_loader, model, device)
    print(f"Test MSE {test_mse:.6f}, MAE {test_mae:.6f}")

    #plotting data collection
    y_true, y_pred = prediction_collecter_plot(test_loader, model, device)

    #plots - save or show option
    plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, save_plot=False, show_plot=False)
    plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=False)
    plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=False, y_limit=0.025, zoom='0.025')


    plot_dataset_vs_learned_marginal(model, device, test_loader, save_plot=False, show_plot=False)
    # plot_flow_posterior_one_example(model, device, test_loader, index_in_batch=1, num_samples=1000, bins=100, save_plot=False,  show_plot=True)
    # plot_uncertainty_vs_error(model, device, test_loader, num_samples=1000, save_plot=False, show_plot=True)

    #idx from [0, len(batch)]
    for idx in range(0, 200, 25):
        plot_flow_posterior_one_example(model, device, test_loader, global_index=idx, show_plot=True)

    plot_error_vs_true_omega(y_true, y_pred, save_plot=False, show_plot=True)



#main loop
def main3(dataset: str="linear"):
    #setup for reproducibility and device
    set_seed()
    device = set_device()

    #setup for model and more
    model = TransformerModel3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    #creating dataset and converting to tensor dataset
    if dataset == "linear":
        V_np, tar_np, t_np = make_double_sine_dataset(noise=True)
    elif dataset == "nonlinear":
        V_np, tar_np, t_np = make_double_sine_nonlinear_dataset(noise=True)
    ds_full = from_array_to_tensor_dataset(V_np, tar_np)

    #splitting to train, val and test parts
    train_loader, val_loader, test_loader = split_and_load(ds_full)

    #training and validation steps
    model, train_mse_hist, val_mse_hist = train_and_eval_training_flow2(train_loader, val_loader, device, model, optimizer, scheduler)

    #test step
    test_mse, test_mae = evaluate2w(test_loader, model, device)
    print(f"Test MSE {test_mse:.6f}, MAE {test_mae:.6f}")

    #plotting data collection
    #y_true, y_pred = prediction_collecter_plot_2w(test_loader, model, device)

    #plots - save or show option
    #plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=False)
    #plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=False, y_limit=0.025, zoom='0.025')

    #plot_pred_vs_true_double(y_true, y_pred, test_mse=test_mse, test_mae=test_mae, save_plot=False, show_plot=True)
    #plot_freq_space_true_vs_pred(y_true, y_pred, test_mse=test_mse, test_mae=test_mae, save_plot=False, show_plot=True)

    #for idx in range(0, 5, 2):
    #    plot_flow_posterior_double_example(model, device, test_loader, global_index=idx, num_samples=100000, bins=50, save_plot=False, show_plot=True)

    
#main1()
#main2()
#main3()
