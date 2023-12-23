# _*_ coding : utf-8 _*_
# @Time : 2023/12/23 20:03
# @Author : 娄星华
# @File : LoraSAM_Model
# @Project : SAM


import os

from torch import nn
from torchvision import transforms
from tqdm import tqdm

import lib
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from statistics import mean

# from SAM_loRA_ImageEncoder import LoraSam
from Dataset import MedicalDataset, get_dataloader
from SAM_loRA_ImageEncoder_MaskDecoder import LoraSam
from torch.nn.functional import threshold, normalize


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 144 / 255, 30 / 255, 0.6])
    H, W = mask.shape[-2:]
    mask_image = mask.reshape(H, W, 1) * color.reshape((1, 1, -1))
    ax.imshow(mask_image)


def show_box(Box, ax):
    x0, y0 = Box[0], Box[1]
    W, H = Box[2] - Box[0], Box[3] - Box[1]
    ax.add_patch(plt.Rectangle((x0, y0), W, H, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":
    # 微调版本

    # 配置信息
    model_type = lib.MODEL_TYPE
    checkpoint = lib.SAM_MODEL
    device = lib.DEVICE

    from segment_anything import SamPredictor, sam_model_registry

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)

    # sam_model.to(device=device)
    # print(sam_model)
    lora_sam = LoraSam(sam_model, 4)
    lora_sam.sam = lora_sam.sam.to(device)

    medical_dataset = MedicalDataset(is_train=0, transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader, val_dataloader = get_dataloader()

    lr = 1e-4
    wd = 0
    optimizer = torch.optim.Adam(filter(lambda P: P.requires_grad, lora_sam.sam.parameters()), lr=lr, weight_decay=wd)

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCELoss()
    num_epochs = lib.EPOCHS
    losses = []

    lora_sam.sam.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        for idx, (input_image, gt_binary_mask, box_torch, original_image_size, input_size) in enumerate(
                train_dataloader):
            # print(idx, (input_image, gt_binary_mask,box_torch))
            # print(input_image.shape, type(input_image))  # torch.Size([1, 3, 1024, 1024]) <class 'torch.Tensor'>

            image_embedding = lora_sam.sam.image_encoder(input_image)

            # print("image:",image_embedding.shape)

            # 锚框
            sparse_embeddings, dense_embeddings = lora_sam.sam.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            print("box:", sparse_embeddings, dense_embeddings)

            # decoder
            low_res_masks, iou_predictions = lora_sam.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=lora_sam.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = lora_sam.sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(
                device=device)

            # print(low_res_masks, low_res_masks.shape)
            # print(upscaled_masks, upscaled_masks.shape)

            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            loss = loss_fn(binary_mask, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(epoch_losses)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

    # 绘制损失曲线
    mean_losses = [mean(x) for x in losses]

    plt.plot(list(range(len(mean_losses))), mean_losses)
    plt.title('Mean epoch loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')

    plt.show()
