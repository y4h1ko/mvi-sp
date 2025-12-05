from .imports_and_libraries import *



def set_seed(seed=cfg.seed, seed_torch=True):
    '''Set seed for reproducibility for everything (random, numpy, torch). Random seed is not set here.'''
    rd.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #for now i print seed
    #print(f'Seed: {cfg.seed}')

def from_array_to_tensor_dataset(V_np_arr: np.ndarray, tar_np_arr: np.ndarray):
    '''Convert NumPy arrays of inputs and targets to a TensorDataset.'''

    #np.arrays to tensors
    X = torch.from_numpy(V_np_arr).float()
    y = torch.from_numpy(tar_np_arr).float()

    return TensorDataset(X, y)

#sine dataset generator
def make_sine_dataset(N: int=cfg.num_of_samples, t_interval: list=cfg.time_interval, t_k: int=cfg.discr_of_time, w_min: float=cfg.omega_min, w_max: float=cfg.omega_max, 
                      seed=cfg.seed, mu: float=cfg.mu, sigma: float=cfg.noise_std, noise: bool=False):
    '''
    Generate a dataset with N samples of sine waves y=sin(w*t) with frequencies w sampled uniformly from [w_min, w_max] - both limit values are included.
    Optionally, Gaussian noise N(mu, sigma) can be added to each sample.

    Returns: tuple of NumPy arrays (V, target, t_val)
    '''

    rng = np.random.default_rng(seed)

    #evenly space time in t_k levels
    t_val = np.linspace(t_interval[0], t_interval[1], t_k)
    #generate N frequencies from [w_min ,w_max]
    w_val = rng.uniform(w_min, w_max, size=N)
    
    V, target = [], []

    #generate N samples of sin(w*t)
    for wi in w_val:
        if noise:
            Vi = 1 * np.sin(wi * t_val) + rng.normal(mu, sigma, size=t_k)
        else:
            Vi = 1 * np.sin(wi * t_val)
        V.append(np.array(Vi))
        target.append(np.array([wi]))

    #return tuple of arrays of 
    return np.array(V), np.array(target), t_val


def make_double_sine_dataset(N: int=cfg.num_of_samples, t_interval: list=cfg.time_interval, t_k: int=cfg.discr_of_time, w_min: float=cfg.omega_min, w_max: float=cfg.omega_max, 
                      seed=cfg.seed, mu: float=cfg.mu, sigma: float=cfg.noise_std, noise: bool=False):
    '''
    Generate a dataset with N samples of sine waves y=sin(wi*t) + sin(wj*t) with frequencies w sampled uniformly from [w_min, w_max] - both limit values are included.
    Optionally, Gaussian noise N(mu, sigma) can be added to each sample.

    Returns: tuple of NumPy arrays (V, target, t_val)
    '''

    rng = np.random.default_rng(seed)

    #evenly space time in t_k levels
    t_val = np.linspace(t_interval[0], t_interval[1], t_k)
    #generate N frequencies from [w_min ,w_max] for two mix omegas
    w_val = rng.uniform(w_min, w_max, size=(N,2))
    
    V, target = [], []

    #generate N samples of sin(wi*t) + sin(wj*t)
    for w1, w2 in w_val:
        if noise:
            Vi = 1 * np.sin(w1 * t_val) + 1 * np.sin(w2 * t_val) + rng.normal(mu, sigma, size=t_k)
        else:
            Vi = 1 * np.sin(w1 * t_val) + 1 * np.sin(w2 * t_val)
        V.append(np.array(Vi))
        
        #target.append(np.sort([w1,w2]))
        target.append(np.array([w1,w2]))
        
    #return tuple of arrays of 
    return np.array(V), np.array(target), t_val

#make_sine_dataset(noise=True)
#make_double_sine_dataset(noise=True)


