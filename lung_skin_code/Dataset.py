# _*_ coding : utf-8 _*_
# @Time : 2023/12/23 17:12
# @Author : 娄星华
# @File : Dataset
# @Project : SAM
import os
import cv2
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

import lib


class MedicalDataset(Dataset):
    def __init__(self, is_train=1, transform=None):
        self.total_name = []
        self.data_path = None
        self.gt_data_path = None
        self.transform = transform
        self.is_train = is_train
        self.image_size = 1024

        if self.is_train:
            self.data_path = lib.image_dirPath
            self.gt_data_path = lib.ground_truth_dir
        else:
            self.data_path = lib.test_image_dirPath
            self.gt_data_path = lib.test_ground_truth_dir

        filenameList = os.listdir(self.data_path)

        self.total_name = [filename[:-4] for filename in filenameList]

        gt_filenameList = os.listdir(self.gt_data_path)

        self.gt_total_name = [filename[:-4] for filename in gt_filenameList]

        self.extension_name = os.path.splitext(filenameList[0])[1]

        self.gt_extension_name = os.path.splitext(gt_filenameList[0])[1]

    def __getitem__(self, index):
        while True:
            image_name = self.total_name[index] + self.extension_name
            image_path = os.path.join(self.data_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Original_image_size = image.shape[:2]  # 原始图片的大小

            transform = ResizeLongestSide(self.image_size)
            Input_image = transform.apply_image(image)

            input_image_torch = torch.as_tensor(Input_image, dtype=torch.float).to(device=lib.DEVICE)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()
            Input_image = self.Pre_process(transformed_image).to(device=lib.DEVICE)  # 输入图片

            # original_image_size = (1024, 1024)
            # input_size = (1024, 1024)

            gt_index = None

            try:
                gt_index = self.gt_total_name.index(self.total_name[index])  # CT DATA

            except ValueError:
                print(f"{self.total_name[index]} not found in the list.")

            gt_image_name = self.gt_total_name[gt_index] + self.gt_extension_name
            gt_image_path = os.path.join(self.gt_data_path, gt_image_name)

            # gt_mask
            unscaled_gt_gray = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)  # RGB加权单通道灰度图读取
            n_Box_torch = self.get_Box(unscaled_gt_gray, Original_image_size, transform)  # n_box

            transform_gt_gray = transform.apply_image(unscaled_gt_gray)
            transform_gt_gray_torch = torch.as_tensor(transform_gt_gray, dtype=torch.float).to(device=lib.DEVICE)

            Input_gt = self.Pre_process(transform_gt_gray_torch.unsqueeze(0).contiguous())
            Input_gt = (Input_gt == 255)  # 转换成布尔矩阵
            Input_gt = torch.as_tensor(Input_gt > 0, dtype=torch.float)

            # Check if any of the variables is empty or None
            if Input_image is None or Input_gt is None or n_Box_torch is None or Original_image_size is None:
                # print("Skipping due to empty or None variable.")
                index = (index + 1) % len(self.total_name)
                continue  # Skip to the next index if any variable is None

            return Input_image, Input_gt, n_Box_torch, Original_image_size

    def __len__(self):
        return len(self.total_name)

    @staticmethod
    def get_Box(Input_gt, Original_image_size, transform):
        n_Box_torch = None
        # print("Input_gt.shape:", Input_gt.shape)
        # print("Input_gt.min(), Input_gt.max():", Input_gt.min(), Input_gt.max())
        contours, hierarchy = cv2.findContours(Input_gt,
                                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓信息
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 对轮廓按照面积从大到小进行排序
        # print(image_name, len(contours))
        n_box = 2  # 想框出来的box数量
        if len(sorted_contours) >= n_box:  # 对遮罩图进行检测，仅有一个有效目标
            Box_torch = [torch.Tensor(0)] * n_box
            for n in range(n_box):
                x, y, w, h = cv2.boundingRect(sorted_contours[n])
                bbox_coord = np.array([x, y, x + w, y + h])
                box = transform.apply_boxes(bbox_coord, Original_image_size)  # scale:1024
                Box_torch[n] = torch.as_tensor(box, dtype=torch.float).to(lib.DEVICE)
            n_Box_torch = torch.cat(Box_torch, dim=0)
        return n_Box_torch

    def Pre_process(self, x: torch.Tensor, flag=0) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""

        if flag:
            # Normalize colors
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).to(lib.DEVICE)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(lib.DEVICE)
            x = (x - pixel_mean.view(3, 1, 1)) / pixel_std.view(3, 1, 1)

        # print(x)
        # Pad
        h, w = x.shape[-2:]
        # print(h, w)
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def get_dataloader(is_train=1, transform=None):
    """
    实现数据集的加载

    :return: 训练集、验证集
    """
    if is_train:
        Medical_dataset = MedicalDataset(is_train=is_train, transform=transform)
        total_size = len(Medical_dataset)
        n = 0.8  # 划分比例
        train_size = int(n * total_size)
        val_size = total_size - train_size

        torch.manual_seed(666)  # 设置随机种子保证每次划分数据集相同
        __train_dataset, __val_dataset = random_split(Medical_dataset, [train_size, val_size])

        __train_dataloader = DataLoader(__train_dataset, batch_size=lib.BATCH_SIZE, shuffle=True,
                                        num_workers=lib.NUM_WORKERS,
                                        drop_last=True)
        __val_dataloader = DataLoader(__val_dataset, batch_size=lib.BATCH_SIZE, shuffle=False,
                                      num_workers=lib.NUM_WORKERS,
                                      drop_last=True)

        return __train_dataloader, __val_dataloader
    else:
        Medical_dataset = MedicalDataset(is_train=is_train, transform=transform)
        __test_dataloader = DataLoader(Medical_dataset, batch_size=lib.BATCH_SIZE, shuffle=False,
                                       num_workers=lib.NUM_WORKERS,
                                       drop_last=True)
        return __test_dataloader


if __name__ == "__main__":

    train_dataloader, val_dataloader = get_dataloader(is_train=1)
    for idx, (input_image, gt_binary_mask, box_torch, original_image_size) in enumerate(train_dataloader):
        # print(input_image.shape, gt_binary_mask.shape, type(input_image), type(gt_binary_mask))
        print(idx, input_image.shape, gt_binary_mask.shape, box_torch, original_image_size)
        # break
