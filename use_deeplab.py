
from YelloHand.checkpoint import load_checkpoint
from YelloHand.models.deeplab import DeepLab
import torch

def load_deeplab(path: str, device=torch.device("cpu")):
    deeplab = DeepLab()
    checkpoint = load_checkpoint(path, device)
    deeplab.load_state_dict(checkpoint["model"])
    deeplab.to(device)
    deeplab.eval()
    return deeplab


