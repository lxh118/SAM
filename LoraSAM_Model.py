from torch import nn
from torchvision import transforms
from tqdm import tqdm

import os
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

from statistics import mean
from torch.nn.functional import threshold, normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from SAM_loRA_ImageEncoder import LoraSam
from SAM_loRA_ImageEncoder_MaskDecoder import LoraSam
# Set up predictors for both tuned and original models
from segment_anything import SamPredictor

from utils import CombinedLoss, show_mask, show_box, dice_coefficient
from Dataset import get_dataloader
import lib

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
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss



    # 微调版本
if __name__ == "__main__":
   
    # 测试文件保存位置
    save_path = './Image_lung_1e-4_maskdc'
    # 判断文件夹是否存在
    if not os.path.exists(save_path):
        # 如果不存在，则创建文件夹
        os.makedirs(save_path)
    # 配置信息
    model_type = lib.MODEL_TYPE
    checkpoint = lib.SAM_MODEL
    device = lib.DEVICE

    from segment_anything import SamPredictor, sam_model_registry

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)

    # sam_model.to(device=device)
    # print(sam_model)1
    lora_sam = LoraSam(sam_model, 8)
    lora_sam.sam = lora_sam.sam.to(device)

    # lora_sam.sam.image_encoder = nn.DataParallel(lora_sam.sam.image_encoder)
    # lora_sam.sam.prompt_encoder = nn.DataParallel(lora_sam.sam.prompt_encoder)
    # lora_sam.sam.mask_decoder = nn.DataParallel(lora_sam.sam.mask_decoder)
    # image_pe=lora_sam.sam.prompt_encoder.module.get_dense_pe(),

    train_dataloader, val_dataloader = get_dataloader(is_train=1, transform=transforms.Compose([transforms.ToTensor()]))

    lr = 1e-4
    wd = 0
    
    optimizer = torch.optim.Adam(filter(lambda P: P.requires_grad, lora_sam.sam.parameters()), lr=lr, 
                            betas=(0.9, 0.999), eps=1e-08, weight_decay=wd, amsgrad=True)
    schedule = ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.3,patience=2)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    num_epochs = lib.EPOCHS
    train_losses = []
    val_losses = []
    iou_res = []
    lora_sam.sam.train()

    # 实例化早停类
    early_stopping = EarlyStopping(patience=2, verbose=True)

    # 开始训练 
    lora_sam.sam.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_Train_loss = []

        for input_image, gt_binary_mask, box_torch, original_image_size in tqdm(train_dataloader):
            # print(idx, (input_image, gt_binary_mask,box_torch))
            # print(input_image.shape, type(input_image))  # torch.Size([1, 3, 1024, 1024]) <class 'torch.Tensor'>
            
            # 将 PyTorch 张量的列表转换为元组列表
            original_image_size = [(t[0].item(), t[1].item()) for t in zip(*original_image_size)]
            image_embedding = lora_sam.sam.image_encoder(input_image)

            # print("image:",image_embedding.shape)

             # print("image:",image_embedding.shape)
            train_outputs = []
            for i, curr_embedding in enumerate(image_embedding):
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
                    original_size=original_image_size[i]).to(device=device)


                unscaled_masks = unscaled_masks[0]+unscaled_masks[1]
                binary_mask = normalize(threshold(unscaled_masks, 0.0, 0))  # 小于0的置零
                train_outputs.append(binary_mask)
                # train_outputs.append({
                #     "masks": binary_mask,
                #     "iou_predictions": iou_predictions,
                #     "low_res_logits": low_res_masks,
                # })

            # Instantiate the combined loss
            combined_loss = CombinedLoss(alpha=0.5, beta=0.5)

            # Calculate the loss
            loss = combined_loss(torch.stack(train_outputs, dim=0), gt_binary_mask)
            # print(loss.item())

            # loss = loss_fn(binary_mask, gt_binary_mask)
            #print("binary_mask, gt_binary_mask",binary_mask.shape, gt_binary_mask.shape,binary_mask,gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_Train_loss.append(loss.item())
        train_losses.append(epoch_Train_loss)
        iou_res.append(iou_predictions)
        # print(iou_res)
        # # 绘制损失曲线
        # plt.plot(list(range(len(train_losses[-1]))), train_losses[-1])
        # plt.title('Batch loss')
        # plt.xlabel('Batch Divided')
        # plt.ylabel('Loss')
        # plt.savefig(os.path.join(save_path, f"{epoch}Batchloss"))


        with torch.no_grad():
            lora_sam.sam.eval()
            epoch_val_loss = []
            for input_image, gt_binary_mask, box_torch, original_image_size in tqdm(val_dataloader):
                # print(idx, (input_image, gt_binary_mask,box_torch))
                # print(input_image.shape, type(input_image))  # torch.Size([1, 3, 1024, 1024]) <class 'torch.Tensor'>
                
                original_image_size = [(t[0].item(), t[1].item()) for t in zip(*original_image_size)]
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
                                                                    original_image_size[i]).to(device=device)
                    unscaled_masks = unscaled_masks[0]+unscaled_masks[1]
                    binary_mask = normalize(threshold(unscaled_masks, 0.0, 0))

                    val_outputs.append(binary_mask)

                combined_loss = CombinedLoss(alpha=0.5, beta=0.5)

            # Calculate the loss
                loss = combined_loss(torch.stack(val_outputs, dim=0), gt_binary_mask)
                # print(loss.item())

                # loss = loss_fn(binary_mask, gt_binary_mask)

                epoch_val_loss.append(loss.item())
        val_losses.append(epoch_val_loss)

        # 在训练结束时调用学习率衰减和早停
        schedule.step(mean(epoch_val_loss))
        early_stopping(mean(epoch_val_loss), lora_sam.sam)

        if early_stopping.early_stop:
            print("早停...")
            break

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
    plt.savefig(os.path.join(save_path, "Epochloss"))

    # # # # 保存模型到文件
    import pickle
    # model_filename = "model_CT_lr1e-3.pkl"
    model_filename = "model_CT_lrd1e-4_maskdc.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(lora_sam.sam, model_file)

    #从文件加载模型
    with open(model_filename, "rb") as model_file:
        loaded_model = pickle.load(model_file)

    # compare our tuned model to the original model
    # Load up the model with default weights
    sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model_orig.to(device)



    predictor_tuned = SamPredictor(loaded_model)
    predictor_original = SamPredictor(sam_model_orig)

    ground_truth_dir = lib.test_ground_truth_dir_CT
    ground_truth_list = os.listdir(ground_truth_dir)

    bbox_coords = {}  # {"image_name":[x, y, x + w, y + h]}
    bbox_coords2 = {}
    ground_truth_masks = {}  # {"image_name":[]}
    # k = len("_Segmentation.png")  # 无用后缀长度
    
    for ground_truth in ground_truth_list:
        f_path = os.path.join(ground_truth_dir, ground_truth)

        gt_grayscale = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)  # RGB加权单通道灰度图读取
        #print(gt_grayscale)
        
        contours, hierarchy = cv2.findContours(gt_grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓信息

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 对轮廓按照面积从大到小进行排序

        # print(image_name, len(contours))

        n_box = 2  # 想框出来的box数量
        if len(sorted_contours) >= n_box:  # 对遮罩图进行检测，仅有一个有效目标
            Box_torch = [torch.Tensor(0)] * n_box
            for n in range(n_box):
                x, y, w, h = cv2.boundingRect(sorted_contours[n])
                if n ==0 :
                    bbox_coords[ground_truth] = np.array([x, y, x + w, y + h])
                else:
                    bbox_coords2[ground_truth] = np.array([x, y, x + w, y + h])
        ground_truth_masks[ground_truth] = (gt_grayscale == 255)  # 转换成布尔矩阵

        # break

    print(ground_truth_masks)
    print(bbox_coords)

    acc1=[]
    acc2=[]
    keys = list(bbox_coords.keys())
    image_dirPath = lib.test_image_dirPath_CT
    for k in keys:
        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (
        1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(lib.DEVICE)

        Gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float)
        image = cv2.imread(os.path.join(image_dirPath, "{0}".format(k)))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor_tuned.set_image(image)
        predictor_original.set_image(image)

        # 使用放缩的框进行predict 使用原始框画图，这里特别注意
        input_bbox = torch.stack([torch.tensor(bbox_coords[k]), torch.tensor(bbox_coords2[k])]).to(device)
        transform_bbox = predictor_original.transform.apply_boxes_torch(input_bbox,image.shape[:2])

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

        masks_tune = masks_tuned[0] + masks_tuned[1]
        masks_ori = masks_orig[0] + masks_orig[1]
        print("masks_tuned", masks_tune.shape, type(masks_tune))
        print("masks_orig", masks_ori.shape, type(masks_ori))

        _, axs = plt.subplots(1, 2, figsize=(25, 25))

        axs[0].imshow(image)
        show_mask(masks_tune.cpu(), axs[0])
        show_box(input_bbox[0].cpu(), axs[0])
        show_box(input_bbox[1].cpu(), axs[0])
        axs[0].set_title('Mask with Tuned Model', fontsize=26)
        axs[0].axis('off')

        axs[1].imshow(image)
        show_mask(masks_ori.cpu(), axs[1])
        show_box(input_bbox[0].cpu(), axs[1])
        show_box(input_bbox[1].cpu(), axs[1])
        axs[1].set_title('Mask with Untuned Model', fontsize=26)
        axs[1].axis('off')

        plt.savefig(os.path.join(save_path, str(k)))

        dc1= dice_coefficient(masks_ori.cpu(),Gt_binary_mask.cpu())
        dc2 = dice_coefficient(masks_tune.cpu(),Gt_binary_mask.cpu())

        # hd1 = hausdorff_distance(masks_ori.cpu().numpy().squeeze().flatten(),Gt_binary_mask.cpu().numpy().squeeze().flatten())
        # hd2 = hausdorff_distance(masks_tune.cpu().numpy().squeeze().flatten(),Gt_binary_mask.cpu().numpy().squeeze().flatten())
        acc1.append(dc1)
        acc2.append(dc2)
        print(dc1,dc2)
    print("原始模型平均准确度：{0}，微调模型平均准确度：{1}".format(np.mean(acc1),np.mean(acc2)))

    pass