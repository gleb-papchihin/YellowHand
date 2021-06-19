import torch


def get_device():
    """
    Prefer to GPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def optimizer_to(optimizer, device):
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def create_checkpoint(path, model, optimizer, loss_fn):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_fn': loss_fn.state_dict(),
    }
    torch.save(to_save, path)

def save_model(path, model):
    to_save = {
        'model': model.state_dict(),
    }
    torch.save(to_save, path)

def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
