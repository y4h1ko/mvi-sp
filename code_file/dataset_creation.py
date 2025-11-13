from .imports_and_libraries import *



def set_seed(seed=SEED, seed_torch=True):
    "Set seed for reproducibility"
    rd.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #for now i print seed
    #print(f'Seed: {seed}')

def from_array_to_tensor_dataset(V_np_arr: np.ndarray, tar_np_arr: np.ndarray):
    #np.arrays to tensors
    X = torch.from_numpy(V_np_arr).float()
    y = torch.from_numpy(tar_np_arr).float()

    return TensorDataset(X, y)

#sine dataset generator
def make_sine_dataset(N: int=NUM_OF_SAMPLES, t_interval: list=TIME_INTERVAL, t_k: int=DISCR_OF_TIME, w_min: float=OMEGA_MIN, w_max: float=OMEGA_MAX, seed=SEED ):
    rng = np.random.default_rng(seed)

    #evenly space time in t_k levels
    t_val = np.linspace(t_interval[0], t_interval[1], t_k)
    #generate N frequencies from [w_min ,w_max]
    w_val = rng.uniform(w_min, w_max, size=N)
    
    V, target = [], []

    #generate N samples of sin(w*t)
    for wi in w_val:
        Vi = 1 * np.sin(wi * t_val)
        V.append(np.array(Vi))
        target.append(np.array([wi]))

    #return tuple of arrays of 
    return np.array(V), np.array(target), t_val






