from pathlib import Path
import numpy as np


SEED = 177013               #seed for reproducibility
NUM_OF_SAMPLES = 1000       #number of samples in dataset
TIME_INTERVAL = [0.0, 2*np.pi]      #interval for time
DISCR_OF_TIME = 100         #level of time discretization
OMEGA_MIN = 0.5             #minimal value for frequency (omega)
OMEGA_MAX = 10              #maximal value for frequency (omega)
AMPLITUDE_MIN = 1           #minimal value for amplitude (A)
AMPLITUDE_MAX = 1           #maximal value for amplitude (A)

EPOCHS = 160 


PLOTS_DIR = Path("plots/rep03-diff-N-and-t_disc-and-EPOCHS")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


