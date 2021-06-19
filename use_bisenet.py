
from YelloHand.checkpoint import load_checkpoint
from YelloHand.models.bisenet import BiSeNet
import torch

def load_bisenet(path: str, device=torch.device("cpu")):
    bisenet = BiSeNet(n_classes=1, train_mode=False)
    checkpoint = load_checkpoint(path, device)
    bisenet.load_state_dict(checkpoint["model"])
    bisenet.to(device)
    bisenet.eval()
    return bisenet


