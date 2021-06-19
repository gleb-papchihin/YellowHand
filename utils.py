from torchvision import transforms
from pathlib import Path
from PIL import Image

import ZipFile
import shutil
import json


def load_image(path: str, resize: tp.Tuple[int, int], device=torch.device("cpu")):
    image = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()
    resizer = transforms.Resize(resize)
    x = to_tensor(image)
    x = resizer(x)
    x = x.to(device).float()
    x = x.unsqueeze(0)
    return x

def create_history():
    history = {
        "mIOU": [],
        "Loss": [],
        "PixelAccuracy": []
    }
    return history

def create_meta():
    meta = {
        "zip_slice_index": 0,
        "epoch": 0,
        "batch": 0
    }
    return meta

def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file)

def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def is_loadable(path: tp.Optional[str]) -> bool:
    if path is None:
        return False
    if Path(path).exists() is False:
        return False
    return True

def backgrounds_was_loaded(path: str):
    folder = Path(path)
    if not folder.exists():
        return False
    if len(list(folder.glob("*"))) == 0:
        return False
    return True

def create_folder(path):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

def delete_folder(path_to_folder: str):
    folder = Path(path_to_folder)
    if folder.exists() is True:
        shutil.rmtree(folder)

def split_on_batches(paths, batch_size):
    n_batches = len(paths) // batch_size
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        stop = start + batch_size
        batch = paths[start:stop]
        batches.append(batch)
    return batches

def extract_zip(path_to_zip :str,
    slice: tp.Optional[tp.Tuple[int, int]]=None) -> None:
    with ZipFile(path_to_zip, 'r') as zip:
        namelist = zip.namelist()
        if slice is None:
            start, stop = (0, len(namelist))
        else:
            start, stop = slice
        zip.extractall(members=namelist[start: stop])
