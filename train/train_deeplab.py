import torch.utils.model_zoo as modelzoo
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
from zipfile import ZipFile
from pathlib import Path
from PIL import Image
from torch import nn

import typing as tp
import shutil
import random
import torch
import json
import os


def fit(batches, preprocessor, model, optimizer, loss_fn, epochs,
    ckpt_path:tp.Optional[str]=None, ckpt_per_iter:tp.Optional[int]=None,
    verbose_per_iter:tp.Optional[int]=None, history_path:tp.Optional[str]=None,
    meta_path: tp.Optional[str]=None):

    n_batches = len(batches)

    # GPU is preferred
    device = get_device()

    # Load model from checkpoint
    if is_loadable(ckpt_path):
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        print("Checkpoint was found")
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        print()
        checkpoint = load_checkpoint(ckpt_path, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])

    # Load history
    if is_loadable(history_path):
        history = load_json(history_path)
    elif history_path is None:
        history = create_history()
        history_path = "history.json"
    else:
        history = create_history()

    # Load meta
    if is_loadable(meta_path):
        meta = load_json(meta_path)
    elif history_path is None:
        meta = create_meta()
        meta_path = "meta.json"
    else:
        meta = create_meta()

    # Read parameters
    start_epoch = meta["epoch"]
    start_batch = meta["batch"]

    # Switch to device
    model.to(device)
    optimizer_to(optimizer, device)

    for epoch in range(start_epoch, epochs):

        local_loss = []
        local_miou = []
        local_mpa = []

        for i in range(start_batch, len(batches)):
            
            batch = batches[i]

            # Data
            torch.cuda.empty_cache()
            model.train()
            loaded = Loader(batch)
            x, y = preprocessor(loaded[:])
            x = x.to(device).float()
            y = y.to(device).float()

            # Step
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            miou = get_mean_iou(pred, y)
            mpa = get_mean_pixel_acc(pred, y)

            # Add to local buffer
            local_loss.append(loss.item())
            local_miou.append(miou)
            local_mpa.append(mpa)

            # Save to dict
            history['Loss'].append(loss.item())
            history['PixelAccuracy'].append(mpa)
            history['mIOU'].append(miou)

            # change meta batch
            meta["batch"] += 1

            # Print state of a training
            if verbose_per_iter is not None:
                if (i+1) % verbose_per_iter == 0:
                    show_state(epoch, i, local_loss, local_miou, local_mpa)
                    local_loss = []
                    local_miou = []
                    local_mpa = []

            # Create checkpoint
            if (ckpt_path is not None) and (ckpt_per_iter is not None):
                if (i+1) % ckpt_per_iter == 0:
                    save_json(meta_path, meta)
                    save_json(history_path, history)
                    create_checkpoint(ckpt_path, model, optimizer, loss_fn)

        # change meta epoch
        meta["epoch"] += 1
        meta["batch"] = 0
        start_batch = 0

if __name__ == "__main__":

    ckpt_path = ""
    meta_path = ""
    history_path = ""

    resize = (480, 640)
    model = DeepLab()
    loss_fn = nn.BCEWithLogitsLoss()
    preprocessor = Preprocessor(resize)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    meta = load_json(meta_path)
    for i in range(meta["epoch"], 5):
        if os.path.exists("Ego2Hand"):
            shutil.rmtree("Ego2Hand")
        extract_zip(subset_0_4, [8000*i, 8000*(i+1)])
        extract_zip(subset_5_10, [8000*i, 8000*(i+1)])
        extract_zip(subset_11_16, [8000*i, 8000*(i+1)])
        extract_zip(subset_17_21, [8000*i, 8000*(i+1)])
        
        gtea_paths = GTEAPaths("GTEA")
        ego_hand_paths = EgoHandPaths("EgoHand")
        ego_2_hand_paths = Ego2HandPaths("Ego2Hand", "backgrounds")
        paths = AugmentationPaths([*gtea_paths[:], *ego_hand_paths[:], *ego_2_hand_paths[:]])
        batches = split_on_batches(paths, 8)

        fit(batches, preprocessor, model, optimizer, loss_fn, (i+1), 
            ckpt_path, 25, 25, history_path, meta_path)