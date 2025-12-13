from pathlib import Path
import numpy as np


'''
This is the GLOBAL configuration file where all hyperparameters and paths are set. Be careful when changing them.
 - SEED is hardwritten for reproducibility.
 - number of samples in dataset: num_of_samples
 - time interval for sine waves: time_interval
 - discretization of time interval: discr_of_time
 - frequency range: omega_min, omega_max
 - noise standard deviation: noise_std

 - model hyperparameters:
    - dimension of internal model representation: dmodel
    - number of self-attention heads: nhead
    - number of layers in model: num_layers
    - dimension of feedforward network: dim_f
    - dropout rate: dropout
    - learning rate for optimizer: learning_rate
    - weight decay for optimizer: weight_decay
    - number of training epochs: epochs

- paths for saving plots and csv files
'''


PLOTS_DIR = Path("plots/rep08-double-better")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = PLOTS_DIR / "T3_hyperprams_linerDoubleWave.csv"


class Config:
    '''Almighty GLOBAL configuration'''

    def __init__(self):
        
        self.seed = 177013                      #seed for reproducibility
        self.num_of_samples = 1000              #number of samples in dataset
        self.time_interval = [0.0, 2*np.pi]     #interval for time
        self.discr_of_time = 100                #level of time discretization
        self.omega_min = 0.5                    #minimal value for frequency (omega)
        self.omega_max = 10                     #maximal value for frequency (omega)

        #N=10000 and t= 100 or 300 for best accuraccy

        #mu should stay zero so there is no any shift
        self.mu = 0.0                   #mean - noise
        self.noise_std = 0.1            #standard deviation - noise
    
        self.epochs = 160                 #number of training epochs
        self.dmodel = 64                  #dimension of model
        self.nhead = 2                    #number of heads
        self.num_layers = 2               #number of layers
        self.dim_f = 64                   #dimension of feedforward network
        self.dropout = 0.1                #dropout rate
        self.batch_size = 32              #batch size for training
        self.learning_rate = 3e-3         #learning rate
        self.weight_decay = 1e-4          #weight decay for optimizer


        #flowhead params
        self.flow_hidden_features=128        #number of features layers in head with flow
        self.flow_num_layers=4              #number of layers in head with flow


        self.plots_dir = PLOTS_DIR
        self.csv_path = CSV_PATH




#exporting object with all configurations as object with properties
cfg = Config()