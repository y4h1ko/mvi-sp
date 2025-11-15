from code_file.imports_and_libraries import *
from code_file.dataset_creation import *
from code_file.positional_encodings import *
from code_file.models import *
from code_file.train_and_test import *
from code_file.visualizations import *

#main loop
def main():
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
    plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, save_plot=False, show_plot=True)
    plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=True)
    plot_loss_curves(train_mse_hist, val_mse_hist, save_plot=False, show_plot=True, y_limit=0.01)



main()