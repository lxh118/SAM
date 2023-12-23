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

    train_dataloader, val_dataloader = get_dataloader(is_train=1, transform=transforms.Compose([transforms.ToTensor()]))

    lr = 1e-4
    wd = 0
    optimizer = torch.optim.Adam(filter(lambda P: P.requires_grad, lora_sam.sam.parameters()), lr=lr, weight_decay=wd)

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCELoss()
    num_epochs = lib.EPOCHS
    train_losses = []
    val_losses = []
    lora_sam.sam.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_Train_loss = []
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
            epoch_Train_loss.append(loss.item())
        train_losses.append(epoch_Train_loss)

        with torch.no_grad():
            lora_sam.sam.eval()
            epoch_val_loss = []
            for idx, (input_image, gt_binary_mask, box_torch, original_image_size, input_size) in enumerate(
                    val_dataloader):
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
                # print("box:", sparse_embeddings, dense_embeddings)

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

                epoch_val_loss.append(loss.item())
            val_losses.append(epoch_val_loss)

        print(f'Epoch: {epoch}')
        print(f'train Mean loss: {mean(epoch_Train_loss)}')
        print(f'val Mean loss: {mean(epoch_val_loss)}')

    # 绘制损失曲线
    train_mean_losses = [mean(x) for x in train_losses]
    val_mean_losses = [mean(x) for x in val_losses]
    plt.plot(list(range(len(train_mean_losses))), train_mean_losses)
    plt.plot(list(range(len(val_mean_losses))), val_mean_losses)

    plt.title('Mean epoch loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig("Image/loss.png")
    plt.show()

    # compare our tuned model to the original model
    # Load up the model with default weights
    sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model_orig.to(device)

    # Set up predictors for both tuned and original models
    from segment_anything import sam_model_registry, SamPredictor

    predictor_tuned = SamPredictor(lora_sam.sam)
    predictor_original = SamPredictor(sam_model_orig)

    ground_truth_dir = lib.ground_truth_dir
    ground_truth_list = os.listdir(ground_truth_dir)

    bbox_coords = {}  # {"image_name":[x, y, x + w, y + h]}
    ground_truth_masks = {}  # {"image_name":[]}
    k = len("_Segmentation.png")  # 无用后缀长度
    for ground_truth in ground_truth_list:
        f_path = os.path.join(ground_truth_dir, ground_truth)

        gt_grayscale = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)  # RGB加权单通道灰度图读取

        contours, hierarchy = cv2.findContours(gt_grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓信息

        if len(contours) == 1:  # 对遮罩图进行检测，仅有一个有效目标
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox_coords[ground_truth[:- k]] = np.array([x, y, x + w, y + h])

        ground_truth_masks[ground_truth[:- k]] = (gt_grayscale == 255)  # 转换成布尔矩阵

        # break

    # print(ground_truth_masks)
    # print(bbox_coords)
    keys = list(bbox_coords.keys())
    image_dirPath = lib.test_image_dirPath
    for k in keys:
        image = cv2.imread(os.path.join(image_dirPath, "{0}.jpg".format(k)))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor_tuned.set_image(image)
        predictor_original.set_image(image)

        input_bbox = np.array(bbox_coords[k])

        masks_tuned, _, _ = predictor_tuned.predict(
            point_coords=None,
            box=input_bbox,
            multimask_output=False,
        )

        masks_orig, _, _ = predictor_original.predict(
            point_coords=None,
            box=input_bbox,
            multimask_output=False,
        )

        _, axs = plt.subplots(1, 2, figsize=(25, 25))

        axs[0].imshow(image)
        show_mask(masks_tuned, axs[0])
        show_box(input_bbox, axs[0])
        axs[0].set_title('Mask with Tuned Model', fontsize=26)
        axs[0].axis('off')

        axs[1].imshow(image)
        show_mask(masks_orig, axs[1])
        show_box(input_bbox, axs[1])
        axs[1].set_title('Mask with Untuned Model', fontsize=26)
        axs[1].axis('off')

        plt.show()

    pass
