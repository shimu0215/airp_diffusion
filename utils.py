from models import DiffusionEGNN
import torch

def load_model(model_path, device, dtype, num_class):

    model = DiffusionEGNN(number_tokens=None, dim=num_class+1).to(device)

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    return model

def save_dataloader(data_list, save_path, name):
    for i, batch in enumerate(data_list):
        torch.save(batch, f'{save_path}/{name}loader_batch_{i}.pt')

def save_lists(train_list, val_list, test_list):

    torch.save(train_list, '/scratch/wzhao20/airp_saved_data/train_list.pt')
    torch.save(val_list, '/scratch/wzhao20/airp_saved_data//val_list.pt')
    torch.save(test_list, '/scratch/wzhao20/airp_saved_data//test_list.pt')

def load_lists(device):

    train_list = torch.load('/scratch/wzhao20/airp_saved_data/train_list.pt', weights_only=False, map_location=device)
    val_list = torch.load('/scratch/wzhao20/airp_saved_data/val_list.pt', weights_only=False, map_location=device)
    test_list = torch.load('/scratch/wzhao20/airp_saved_data/test_list.pt', weights_only=False, map_location=device)

    return train_list, val_list, test_list

def set_seed(seed=42):
    random.seed(seed)                         # Python
    # np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # CPU上torch
    torch.cuda.manual_seed(seed)              # GPUtorch current
    torch.cuda.manual_seed_all(seed)          # GPUtorch all
    torch.backends.cudnn.deterministic = True # cuDNN
    torch.backends.cudnn.benchmark = False    # cuDNN

def get_optim(lr, model):
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr, amsgrad=True,
        weight_decay=1e-12)

    return optim