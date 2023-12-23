# _*_ coding : utf-8 _*_
# @Time : 2023/12/20 14:08
# @Author : 娄星华
# @File : test.py
# @Project : SAM
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


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
    ground_truth_dir = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_GroundTruth"
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

    print(ground_truth_masks)
    print(bbox_coords)

    # 查看下结果
    # name = "ISIC_0000000"
    #
    # image = cv2.imread(r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data\{0}.jpg"
    #                    .format(name))
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #
    # show_box(bbox_coords[name], plt.gca())
    # show_mask(ground_truth_masks[name], plt.gca())
    # plt.axis('off')
    # plt.show()

    # 配置信息
    # model_type = 'vit_h'
    # checkpoint = 'sam_vit_h_4b8939.pth'

    # curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    model_type = 'vit_h'
    checkpoint = 'sam_vit_h_4b8939.pth'
    device = 'cuda'

    from segment_anything import SamPredictor, sam_model_registry

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device=device)
    print(sam_model)

    # Preprocess the images
    from collections import defaultdict

    from segment_anything.utils.transforms import ResizeLongestSide

    # 对图片做预处理，使其符合输入的格式
    transformed_data = defaultdict(dict)
    # for k in bbox_coords.keys():
    #     image = cv2.imread(r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data\{0}.jpg"
    #                        .format(k))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    #     input_image = transform.apply_image(image)
    #     input_image_torch = torch.as_tensor(input_image).to(device=device)
    #     transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    #
    #     input_image = sam_model.preprocess(transformed_image)
    #     original_image_size = image.shape[:2]
    #     input_size = tuple(transformed_image.shape[-2:])
    #
    #     transformed_data[k]['image'] = input_image
    #     transformed_data[k]['input_size'] = input_size
    #     transformed_data[k]['original_image_size'] = original_image_size

    # Set up the optimizer, hyperparameter tuning will improve performance here
    lr = 1e-4
    wd = 0
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    keys = list(bbox_coords.keys())

    # fine-tuning
    from statistics import mean
    from tqdm import tqdm
    from torch.nn.functional import threshold, normalize

    num_epochs = 20
    losses = []

    sam_model.train()

    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        # Just train on the first 20 examples
        for k in keys[:20]:
            # input_image = transformed_data[k]['image'].to(device=device)
            # input_size = transformed_data[k]['input_size']
            # original_image_size = transformed_data[k]['original_image_size']

            image = cv2.imread(r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data\{0}.jpg"
                               .format(k))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            input_image = transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image).to(device=device)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            input_image = sam_model.preprocess(transformed_image).to(device=device)
            original_image_size = image.shape[:2]
            input_size = tuple(transformed_image.shape[-2:])

            import torch.nn.utils as utils

            # 手动计算并记录梯度范数
            total_norm = 0
            for p in sam_model.mask_decoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Total gradient norm: {total_norm}")  # 若想查看梯度裁剪后的结果可以在后面在计算一次

            # 添加梯度裁剪
            utils.clip_grad_norm_(sam_model.mask_decoder.parameters(), max_norm=1)  # 可以根据需要调整

            # 手动计算并记录梯度范数
            total_norm = 0
            for p in sam_model.mask_decoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Total gradient norm: {total_norm}")

            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                # sam_model.eval()
                image_embedding = sam_model.image_encoder(input_image)

                prompt_box = bbox_coords[k]
                transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                box = transform.apply_boxes(prompt_box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float).to(device=device)
                box_torch = box_torch[None, :]

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(
                device=device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (
                1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device=device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

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

    # compare our tuned model to the original model
    # Load up the model with default weights
    sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model_orig.to(device)

    # Set up predictors for both tuned and original models
    from segment_anything import sam_model_registry, SamPredictor

    predictor_tuned = SamPredictor(sam_model)
    predictor_original = SamPredictor(sam_model_orig)

    # The model has not seen keys[21] (or keys[20]) since we only trained on keys[:20]
    k = keys[10]
    image = cv2.imread(r'D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data\{0}.jpg'
                       .format(k))
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
