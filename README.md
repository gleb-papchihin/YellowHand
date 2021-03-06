# Real time semantic hand segmentation.
### BisenetV2 and DeeplabV3-MobilnetV3.

Two pretrained models for real time segmentation. Ready to use :)

![DeeplabV3](https://github.com/gleb-papchihin/YellowHand/blob/main/DeepLabV3.gif)

# BisenetV2.

| MODEL | IM. SIZE | FPS on GPU | TEST IoU | LINK |
|:-------------:|:---------:|:---------:|:---------:|:------------:|
| BiSeNetV2 | 640x480 | ~237 | 0.94 | [download](https://drive.google.com/file/d/14KR69wtFp8v_DRw96hGZmD84cGK3N0Bv/view) |

# DeeplabV3-MobilnetV3.

| MODEL | IM. SIZE | FPS on GPU | TEST IoU | LINK |
|:-------------:|:---------:|:---------:|:---------:|:------------:|
| DeeplabV3 | 640x480 | ~131 | 0.93 | [download](https://drive.google.com/file/d/1--Kop-l0QLexDlmzQODTnNLtPSHzK8F_/view) |

# Datasets.

| DATASET | SIZE | LINK |
|:---------------------:|:---------:|:------------:|
| Extended GTEA Gaze+ | ~14K | [click](http://cbs.ic.gatech.edu/fpv/) |
| Ego2Hand | ~180K | [click](https://github.com/AlextheEngineer/Ego2Hands) |
| EgoHand | ~5K | [click](http://vision.soic.indiana.edu/projects/egohands/) |

# How to use.

```python
  import torch
  import YellowHand as yellow
  
  # Enter path to model params
  path_to_bisenet = "" 
  path_to_deeplab = "" 
  
  # Select device: cpu or cuda
  device = torch.device("cpu")
  
  # Model was automatically switched to eval mode
  bisenet = yellow.load_bisenet(path_to_bisenet, device)
  deeplab = yellow.load_deeplab(path_to_deeplab, device)
  
  # Have fun
  a = torch.rand([1, 3, 480, 640])
  pred_bisenet = bisenet(a)
  pred_deeplab = deeplab(a)
```
