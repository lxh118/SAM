# _*_ coding : utf-8 _*_
# @Time : 2023/12/23 20:03
# @Author : 娄星华
# @File : LoraSAM_Model
# @Project : SAM

import os
import cv2
import numpy as np
import torch
import pickle
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from statistics import mean
from torch.nn.functional import threshold, normalize
from segment_anything import sam_model_registry, SamPredictor

import lib
from SAM_loRA_ImageEncoder import LoraSam
# from SAM_loRA_ImageEncoder_MaskDecoder import LoraSam
from Dataset import get_dataloader
from MyLoss import CombinedLoss
from DrawImage import show_box, show_mask, Epoch_loss


class EarlyStopping:
    def __init__(self, patience=2, verbose=False, delta=0):
        """
        参数:
        patience (int): 在早停前可以容忍多少个epoch没有改善
        verbose (bool): 是否打印早停信息
        delta (float): “改善”需要超过这个阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """如果验证损失下降，则保存模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'ckpt/checkpoint.pt')
        self.val_loss_min = val_loss


def Train(Train_dataloader, Val_dataloader, Optimizer):
    # Instantiate the combined loss
    loss_fn = CombinedLoss(alpha=0.5, beta=0.5)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()

    lora_sam.sam.train()
    Epoch_Train_loss = []
    for input_image, Gt_binary_mask, box_torch, original_image_size in tqdm(Train_dataloader):

        # print(idx, (input_image, gt_binary_mask,box_torch))
        # print(input_image.shape, type(input_image))  # torch.Size([1, 3, 1024, 1024]) <class 'torch.Tensor'>

        # 将 PyTorch 张量的列表转换为元组列表
        original_image_size = [(t[0].item(), t[1].item()) for t in zip(*original_image_size)]
        image_embedding = lora_sam.sam.image_encoder(input_image)

        # print("image:",image_embedding.shape)
        train_outputs = []
        for i, curr_embedding in enumerate(image_embedding):
            # print(curr_embedding.shape)
            # 锚框
            sparse_embeddings, dense_embeddings = lora_sam.sam.prompt_encoder(
                points=None,
                boxes=box_torch[i],
                masks=None,
            )

            # decoder
            low_res_masks, iou_predictions = lora_sam.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=lora_sam.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            unscaled_masks = lora_sam.sam.postprocess_masks(
                low_res_masks,
                input_size=input_image[i].shape[-2:],
                original_size=original_image_size[i]).to(lib.DEVICE)

            # 在lung数据中，由于我们有两个bbox，故此处的mask有两个unscaled_masks，我们把它加在一起
            unscaled_masks = unscaled_masks[0] + unscaled_masks[1]
            binary_mask = normalize(threshold(unscaled_masks, 0.0, 0))  # 小于0的置零
            train_outputs.append(binary_mask)
            # train_outputs.append({
            #     "masks": binary_mask,
            #     "iou_predictions": iou_predictions,
            #     "low_res_logits": low_res_masks,
            # })

        train_Gt_outputs = PostProcess(Gt_binary_mask, original_image_size)

        # Calculate the loss
        # loss = loss_fn(binary_mask, Gt_binary_mask)
        loss = loss_fn(torch.stack(train_outputs, dim=0), torch.cat(train_Gt_outputs, dim=0))
        # print(loss.item())
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        Epoch_Train_loss.append(loss.item())

    lora_sam.sam.eval()
    Epoch_val_loss = []
    with torch.no_grad():
        for input_image, Gt_binary_mask, box_torch, original_image_size in tqdm(Val_dataloader):
            original_image_size = [(t[0].item(), t[1].item()) for t in zip(*original_image_size)]
            # print(idx, (input_image, gt_binary_mask,box_torch))
            # print(input_image.shape, type(input_image))  # torch.Size([1, 3, 1024, 1024]) <class 'torch.Tensor'>

            image_embedding = lora_sam.sam.image_encoder(input_image)

            # print("image:",image_embedding.shape)
            val_outputs = []
            for i, curr_embedding in enumerate(image_embedding):
                # 锚框
                sparse_embeddings, dense_embeddings = lora_sam.sam.prompt_encoder(
                    points=None,
                    boxes=box_torch[i],
                    masks=None,
                )
                # print("box:", sparse_embeddings, dense_embeddings)

                # decoder
                low_res_masks, iou_predictions = lora_sam.sam.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=lora_sam.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                unscaled_masks = lora_sam.sam.postprocess_masks(low_res_masks, input_image[i].shape[-2:],
                                                                original_image_size[i]).to(lib.DEVICE)
                unscaled_masks = unscaled_masks[0] + unscaled_masks[1]
                binary_mask = normalize(threshold(unscaled_masks, 0.0, 0))

                val_outputs.append(binary_mask)

            val_Gt_outputs = PostProcess(Gt_binary_mask, original_image_size)

            # Calculate the loss
            loss = loss_fn(torch.stack(val_outputs, dim=0), torch.cat(val_Gt_outputs, dim=0))
            # print(loss.item())
            # loss = loss_fn(binary_mask, gt_binary_mask)

            Epoch_val_loss.append(loss.item())

    return Epoch_Train_loss, Epoch_val_loss


def PostProcess(Gt_binary_mask, original_image_size):
    val_Gt_outputs = []
    for i, curr_Gt_binary_mask in enumerate(Gt_binary_mask):
        unscaledGt_binary_mask = lora_sam.sam.postprocess_masks(
            curr_Gt_binary_mask.unsqueeze(0),
            input_size=Gt_binary_mask.shape[-2:],
            original_size=original_image_size[i]).to(lib.DEVICE)
        val_Gt_outputs.append(unscaledGt_binary_mask)
    return val_Gt_outputs


def dice_coefficient(y_true, y_pred):
    """
    Get Acc
    :param y_true:
    :param y_pred:
    :return:
    """
    intersection = np.sum(y_true.numpy().squeeze().flatten() * y_pred.numpy().squeeze().flatten())
    return (2. * intersection) / (
            np.sum(y_true.numpy().squeeze().flatten()) + np.sum(y_pred.numpy().squeeze().flatten()))


def TestImage(Image, Masks_tune, Masks_ori, Input_bbox, K):
    """
    Draw Test Image
    :param Image:
    :param Masks_tune:
    :param Masks_ori:
    :param Input_bbox:
    :param K:
    :return:
    """
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    axs[0].imshow(Image)
    show_mask(Masks_tune.cpu(), axs[0])
    show_box(Input_bbox[0].cpu(), axs[0])
    show_box(Input_bbox[1].cpu(), axs[0])
    axs[0].set_title('Mask with Tuned Model', fontsize=26)
    axs[0].axis('off')
    axs[1].imshow(Image)
    show_mask(Masks_ori.cpu(), axs[1])
    show_box(Input_bbox[0].cpu(), axs[1])
    show_box(Input_bbox[1].cpu(), axs[1])
    axs[1].set_title('Mask with UnTuned Model', fontsize=26)
    axs[1].axis('off')
    plt.savefig(os.path.join(lib.image_savePath, str(K)))


def Test(predictor_tuned, predictor_original, Image, gt_mask, bbox_coord, bbox_coord2):
    """
    :param predictor_original:
    :param predictor_tuned:
    :param Image:
    :param gt_mask:
    :param bbox_coord:
    :param bbox_coord2:
    :return:
    """
    gt_mask_resized = torch.from_numpy(np.resize(gt_mask, (
        1, gt_mask.shape[0], gt_mask.shape[1]))).to(lib.DEVICE)
    Gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float)

    predictor_tuned.set_image(Image)
    predictor_original.set_image(Image)
    # 使用放缩的框进行predict 使用原始框画图，这里特别注意
    Input_bbox = torch.stack([torch.tensor(bbox_coord), torch.tensor(bbox_coord2)]).to(lib.DEVICE)
    transform_bbox = predictor_original.transform.apply_boxes_torch(Input_bbox, Image.shape[:2])
    masks_tuned, _, _ = predictor_tuned.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transform_bbox,
        multimask_output=False,
    )
    masks_orig, _, _ = predictor_original.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transform_bbox,
        multimask_output=False,
    )
    Masks_tune = masks_tuned[0] + masks_tuned[1]
    Masks_ori = masks_orig[0] + masks_orig[1]

    return Gt_binary_mask, Input_bbox, Masks_tune, Masks_ori


def Create_Box_GtMask():
    """
    Create Box and GtMask for Test
    :return: Bbox_coords, Bbox_coords2, Ground_truth_masks
    """
    ground_truth_list = os.listdir(lib.test_ground_truth_dir)
    Bbox_coords = {}  # {"image_name":[x, y, x + w, y + h]}
    Bbox_coords2 = {}
    Ground_truth_masks = {}  # {"image_name":[]}
    for ground_truth in ground_truth_list:
        f_path = os.path.join(lib.test_ground_truth_dir, ground_truth)

        gt_gray = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)  # RGB加权单通道灰度图读取

        contours, hierarchy = cv2.findContours(gt_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓信息
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 对轮廓按照面积从大到小进行排序
        # print(sorted_contours)
        n_box = 2  # 想框出来的box数量
        if len(sorted_contours) >= n_box:  # 对遮罩图进行检测，仅有一个有效目标
            # Box_torch = [torch.Tensor(0)] * n_box
            for n in range(n_box):
                x, y, w, h = cv2.boundingRect(sorted_contours[n])
                if n == 0:
                    Bbox_coords[ground_truth] = np.array([x, y, x + w, y + h])
                else:
                    Bbox_coords2[ground_truth] = np.array([x, y, x + w, y + h])
        Ground_truth_masks[ground_truth] = (gt_gray == 255)  # 转换成布尔矩阵

    return Bbox_coords, Bbox_coords2, Ground_truth_masks


def Trainer():
    train_dataloader, val_dataloader = get_dataloader(is_train=1, transform=transforms.Compose([transforms.ToTensor()]))
    optimizer = torch.optim.Adam(filter(lambda P: P.requires_grad, lora_sam.sam.parameters()), lr=lib.LEARNING_RATE,
                                 weight_decay=lib.WEIGHT_DECAY)
    schedule = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=2, verbose=True)
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=2, verbose=True)  # 实例化早停类
    for epoch in range(lib.EPOCHS):
        print(f'Epoch: {epoch + 1}')
        epoch_Train_loss, epoch_val_loss = Train(train_dataloader, val_dataloader, optimizer)

        # 在训练结束时调用学习率衰减和早停
        schedule.step(mean(epoch_val_loss))
        early_stopping(mean(epoch_val_loss), lora_sam.sam)

        if early_stopping.early_stop:
            print("早停...")
            break

        print(f'train Mean loss: {mean(epoch_Train_loss)}')
        print(f'val Mean loss: {mean(epoch_val_loss)}')

        train_losses.append(epoch_Train_loss)
        val_losses.append(epoch_val_loss)
    # Save Model
    with open(lib.model_fileName, "wb", encoding="utf-8") as Model_file:
        pickle.dump(lora_sam.sam, Model_file)
    # 绘制损失曲线
    Epoch_loss(Train_losses=train_losses, Val_losses=val_losses)


def Tester():
    # 从文件加载模型
    with open(lib.model_fileName, "rb", encoding="utf-8") as model_file:
        loaded_model = pickle.load(model_file)
    # compare our tuned model to the original model
    # Load up the model with default weights
    sam_model_orig = sam_model_registry[lib.MODEL_TYPE](checkpoint=lib.CHECKPOINT)
    sam_model_orig.to(lib.DEVICE)
    predictor_tuned = SamPredictor(loaded_model)
    predictor_original = SamPredictor(sam_model_orig)
    # 创建锚框和遮罩字典
    bbox_coords, bbox_coords2, ground_truth_masks = Create_Box_GtMask()
    print(ground_truth_masks)
    print(bbox_coords)
    acc_ori = []
    acc_tune = []
    for k in list(bbox_coords.keys()):
        image = cv2.imread(os.path.join(lib.test_image_dirPath, "{0}".format(k)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调用模型测试
        gt_binary_mask, input_bbox, masks_tune, masks_ori = Test(predictor_tuned, predictor_original, image,
                                                                 ground_truth_masks[k], bbox_coords[k], bbox_coords2[k])
        # print("masks_tuned", masks_tune.shape, type(masks_tune))
        # print("masks_orig", masks_ori.shape, type(masks_ori))

        # 保存测试图像
        TestImage(image, masks_tune, masks_ori, input_bbox, k)

        dc_ori = dice_coefficient(masks_ori.cpu(), gt_binary_mask.cpu())
        dc_tune = dice_coefficient(masks_tune.cpu(), gt_binary_mask.cpu())

        acc_ori.append(dc_ori)
        acc_tune.append(dc_tune)

        print("{0}: 原始模型平均准确度：{1}，微调模型平均准确度：{2}".format(k, dc_ori, dc_tune))
    print("原始模型平均准确度：{0}，微调模型平均准确度：{1}".format(np.mean(acc_ori), np.mean(acc_tune)))


if __name__ == "__main__":
    sam_model = sam_model_registry[lib.MODEL_TYPE](checkpoint=lib.CHECKPOINT)
    # print(sam_model)

    lora_sam = LoraSam(sam_model, 4)
    lora_sam.sam = lora_sam.sam.to(lib.DEVICE)

    Trainer()

    Tester()
