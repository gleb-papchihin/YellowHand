import typing as tp
import torch


def get_statistics(subset: tp.Tuple[float]) -> tp.Tuple[float, float, float]:
    
    """
    Return median, mean, std for a subset.
    """

    median = torch.tensor(subset).median().item()
    mean = torch.tensor(subset).mean().item()
    std = torch.tensor(subset).std().item()
    return median, mean, std

def show_state(epoch_index: int, batch_index: int, 
    subset_loss: tp.List[float], subset_miou: tp.List[float], 
    subset_mpa: tp.List[float], delimiter: str="*") -> None:
    
    """
    Show value of metrics while training.
    MPA is Mean Pixel accuracy
    MIou is Mean Intersection over union
    """

    loss_median, loss_mean, loss_std = get_statistics(subset_loss)
    miou_median, miou_mean, miou_std = get_statistics(subset_miou)
    mpa_median, mpa_mean, mpa_std = get_statistics(subset_mpa)
    
    print(delimiter, delimiter, delimiter)
    print()
    print(f"Epoch       : {epoch_index}")
    print(f"Batch       : {batch_index}")
    print()
    print(f"LOSS Median : {loss_median}")
    print(f"LOSS Mean   : {loss_mean}")
    print(f"LOSS std    : {loss_std}")
    print()
    print(f"MIoU Median : {miou_median}")
    print(f"MIoU Mean   : {miou_mean}")
    print(f"MIoU Std    : {miou_std}")
    print()
    print(f"MPA Median : {mpa_median}")
    print(f"MPA Mean   : {mpa_mean}")
    print(f"MPA Std    : {mpa_std}")
    print()
