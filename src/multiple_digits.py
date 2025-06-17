import torch
import matplotlib
matplotlib.use('qtagg')
from torch import nn
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./2310545_312579-P8HU6K-966.jpg")

plt.imshow(image)
plt.show()
