
from torchvision import models
from torch import nn

deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, 
                                                           progress=False,
                                                           num_classes=1)

class DeepLab(nn.Module):
  def __init__(self):
    super(DeepLab, self).__init__()
    self.deeplab = deeplab
  def forward(self, x):
    return self.deeplab(x)['out']