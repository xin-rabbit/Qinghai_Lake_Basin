# des: dataset and dataloader for deep learning tasks in remote sensing

import random
import torch
import rasterio as rio
import numpy as np

class crop:
    '''随机裁剪到指定大小 (H, W)，支持图像为 (H, W, C) 或 (C, H, W) 格式'''
    def __init__(self, size=(512, 512)):
        self.size = size  # (h, w)

    def __call__(self, image, truth):
        '''
        image: 可以是 (H, W, C) 或 (C, H, W)
        truth: (H, W)
        返回与输入 image 相同格式的裁剪结果
        '''
        h, w = truth.shape
        start_h = random.randint(0, h - self.size[0])
        start_w = random.randint(0, w - self.size[1])

        # 裁剪图像
        if image.ndim == 3:
            # 假设为 (H, W, C)
            patch = image[start_h:start_h+self.size[0], start_w:start_w+self.size[1], :]
        else:
            # 如果是 (C, H, W)
            patch = image[:, start_h:start_h+self.size[0], start_w:start_w+self.size[1]]

        truth_patch = truth[start_h:start_h+self.size[0], start_w:start_w+self.size[1]]
        return patch, truth_patch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths_scene, paths_truth, augment=True):
        self.paths_scene = paths_scene
        self.paths_truth = paths_truth
        self.augment = augment
        self.crop = crop(size=(512, 512))

    def __getitem__(self, idx):
        # 加载图像和标签
        scene_path = self.paths_scene[idx]
        truth_path = self.paths_truth[idx]
        with rio.open(scene_path) as src, rio.open(truth_path) as truth_src:
            scene_arr = src.read().transpose((1, 2, 0))   # (H, W, C)
            truth_arr = truth_src.read(1)                 # (H, W)

        # 归一化到 [0, 1]
        scene_arr = scene_arr / 10000.0
        scene_arr = scene_arr.astype(np.float32)

        # 随机裁剪到 512x512 (保持 H,W,C 格式)
        scene_crop, truth_crop = self.crop(scene_arr, truth_arr)

        # 数据增强（仅在训练时进行）
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                scene_crop = np.fliplr(scene_crop)
                truth_crop = np.fliplr(truth_crop)

            # 随机垂直翻转
            if random.random() > 0.5:
                scene_crop = np.flipud(scene_crop)
                truth_crop = np.flipud(truth_crop)

            # 随机旋转 (90°, 180°, 270°)
            k = random.choice([0, 1, 2, 3])
            if k != 0:
                scene_crop = np.rot90(scene_crop, k, axes=(0, 1))
                truth_crop = np.rot90(truth_crop, k, axes=(0, 1))

            # 随机亮度调整 (仅图像)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                scene_crop = scene_crop * brightness
                scene_crop = np.clip(scene_crop, 0, 1)

        # 转换为 (C, H, W) 格式
        scene_crop = scene_crop.transpose((2, 0, 1))   # (C, 512, 512)
        truth_crop = truth_crop[np.newaxis, :].astype(np.float32)  # (1, 512, 512)

        # 确保数组内存连续，解决负步长问题（flip/rot90 可能产生负步长视图）
        scene_crop = np.ascontiguousarray(scene_crop)
        truth_crop = np.ascontiguousarray(truth_crop)

        patch = torch.from_numpy(scene_crop).float()
        truth = torch.from_numpy(truth_crop).float()

        return patch, truth

    def __len__(self):
        return len(self.paths_scene)