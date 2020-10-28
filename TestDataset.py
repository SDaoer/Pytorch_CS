import torch
import os
import cv2
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

def img_processor(img_name_list, path, filter_size=33, stride=33, channels=3):
    image_blocks = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    image = image[:, :, 0]
    h, w = image.shape
    h_n = ((h - filter_size) // stride) + 1
    w_n = ((w - filter_size) // stride) + 1

    for i in range(h_n):
        for j in range(w_n):
            block = image[i * stride : (i * stride) + filter_size, j * stride : (j * stride) + filter_size]
            image_blocks.append(block)

    return np.array(image_blocks)

class TestDataset(Dataset):
    def __init__(self, data_dir, mat, transform = None, phi=0.25):
        # 输入的 data_dir 是图片的绝对路径
        self.img_dir = data_dir
        self.transform = transform
        self.image_blocks = img_processor(self.img_dir, data_dir)
        self.phi = phi
        self.mat = mat

    def __len__(self):
        return len(self.image_blocks)

    def __getitem__(self, idx):
        image_block = self.image_blocks[idx]
        label = image_block
        if self.transform is not None:
            image_block = self.transform(image_block)
            label = self.transform(label)
        image_block = image_block.view(33 * 33)
        label = label.view(33 * 33)
        image_block = image_block.double()
        label = label.double()
        # 压缩图像
        with torch.no_grad():
            image_block = torch.matmul(self.mat, image_block)
        return image_block, label