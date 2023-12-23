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
from torchvision import transforms
from segment_anything.modeling import Sam
from torch.nn import functional as F

import lib


def Pre_process(x: torch.Tensor, flag=0) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    if not flag:
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).to(lib.DEVICE)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(lib.DEVICE)
        x = (x - pixel_mean.view(3, 1, 1)) / pixel_std.view(3, 1, 1)
    image_size = 1024
    # print(x)
    # Pad
    h, w = x.shape[-2:]
    # print(h, w)
    padh = image_size - h
    padw = image_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


class MedicalDataset(Dataset):
    def __init__(self, is_train=1, transform=None):
        self.total_name = []
        self.data_path = None
        self.gt_data_path = None
        self.transform = transform
        self.is_train = is_train

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

    def __getitem__(self, index):
        image_name = self.total_name[index] + ".jpg"
        image_path = os.path.join(self.data_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = ResizeLongestSide(1024)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).to(device=lib.DEVICE)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()

        input_image = Pre_process(transformed_image).to(device=lib.DEVICE)
        # original_image_size = image.shape[:2]
        # input_size = tuple(transformed_image.shape[-2:])

        original_image_size = (1024, 1024)
        input_size = (1024, 1024)

        box_torch = None
        # try:
        # k = len("_Segmentation.png")  # 无用后缀长度
        gt_index = self.gt_total_name.index(self.total_name[index] + "_Segmentation")
        gt_image_name = self.gt_total_name[gt_index] + ".png"
        gt_image_path = os.path.join(self.gt_data_path, gt_image_name)

        gt_grayscale = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)  # RGB加权单通道灰度图读取

        contours, hierarchy = cv2.findContours(gt_grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓信息

        if len(contours) == 1:  # 对遮罩图进行检测，仅有一个有效目标
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox_coord = np.array([x, y, x + w, y + h])
            box = transform.apply_boxes(bbox_coord, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float).to(lib.DEVICE)

        gt_grayscale = transform.apply_image(gt_grayscale)

        ground_truth_mask = (gt_grayscale == 255)  # 转换成布尔矩阵

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask, (
            1, ground_truth_mask.shape[0], ground_truth_mask.shape[1]))).to(lib.DEVICE)

        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        gt_binary_mask = Pre_process(gt_binary_mask, flag=1)

        # except ValueError:
        #     print(f"{self.total_name[index]} not found in the list.")

        # print(input_image.shape, gt_binary_mask.shape, type(input_image), type(gt_binary_mask))
        return input_image, gt_binary_mask, box_torch, original_image_size, input_size

    def __len__(self):
        return len(self.total_name)


def get_dataloader():
    """
    实现数据集的加载

    :return: 训练集、验证集
    """
    Medical_dataset = MedicalDataset(is_train=0)
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


if __name__ == "__main__":
    medical_dataset = MedicalDataset(is_train=0, transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader, val_dataloader = get_dataloader()
    for idx, (input_image, gt_binary_mask, box_torch, _, _) in enumerate(train_dataloader):
        # print(input_image.shape, gt_binary_mask.shape, type(input_image), type(gt_binary_mask))
        print(idx, (input_image, gt_binary_mask, box_torch))
        break
