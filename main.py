import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from ultralytics import YOLO
from segment_anything import sam_model_registry


# plt.switch_backend('TkAgg')  # Use TkAgg as the backend


def draw_masks_fromDict(Image, Masks_generated):
    """
    Draw masks on the input image.
    
    :param Image: The original input image.
    :param Masks_generated: A list of dictionaries containing mask information.
    :return: The final image with drawn masks.
    """
    masked_image = Image.copy()

    # Iterate through each mask in masks_generated using enumerate
    for index, mask_info in enumerate(Masks_generated):
        # Use np.repeat to create a mask with the same shape as the image
        Mask = mask_info['segmentation'].astype(int)[:, :, np.newaxis]
        mask_repeated = np.repeat(Mask, 3, axis=2)

        # Use np.random.choice to generate random pixel values for the masked regions
        random_pixels = np.random.choice(range(256), size=3)

        # Use np.where to replace pixels in masked_image where the mask is True
        masked_image = np.where(mask_repeated, random_pixels, masked_image)

        # Convert the resulting image to uint8
        masked_image = masked_image.astype(np.uint8)

    # Use cv2.addWeighted to blend the original image and the image with drawn masks
    return cv2.addWeighted(Image, 0.3, masked_image, 0.7, 0)


def box_label(Image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """
    具体的绘制函数

    :param Image:要绘制边界框和标签的图像。
    :param box:边界框的坐标，格式为 (x_min, y_min, x_max, y_max)。
    :param label:要显示的标签。
    :param color:边界框和标签背景的颜色。
    :param txt_color:标签文本的颜色。
    :return:
    """
    # 计算绘制边界框时的线宽
    lw = max(round(sum(Image.shape) / 2 * 0.003), 2)

    # 边界框的两个顶点坐标
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

    # 绘制边界框
    cv2.rectangle(Image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    # 如果有标签，绘制标签
    if label:
        # 计算字体的粗细
        tf = max(lw - 1, 1)

        # 获取标签文本的宽度和高度
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]

        # 判断标签的位置，以便避免覆盖边界框
        outside = p1[1] - h >= 3

        # 计算标签区域的坐标
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

        # 绘制填充的矩形作为标签的背景
        cv2.rectangle(Image, p1, p2, color, -1, cv2.LINE_AA)

        # 绘制标签文本
        cv2.putText(Image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def plot_bboxes(Image, Boxes, labels=None, colors=None, score=True, conf=None, kkk=0):
    """
    在图像上绘制边界框和标签。

    :param Image:包含目标的图像。
    :param Boxes:包含边界框信息的列表，每个边界框是一个包含坐标和类别信息的数组。
    :param labels:类别标签，默认使用COCO数据集的标签。
    :param colors:每个类别的颜色，默认使用一组预定义的颜色。
    :param score:是否显示边界框上的分数。
    :param conf:置信度阈值，只显示高于阈值的边界框。
    :return:
    """

    if colors is None:
        # Define COCO colors
        colors = [(89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98),
                  (130, 172, 179), (115, 209, 128), (204, 79, 135), (136, 126, 185), (209, 213, 45), (44, 52, 10),
                  (101, 158, 121), (179, 124, 12), (25, 33, 189), (45, 115, 11), (73, 197, 184), (62, 225, 221),
                  (32, 46, 52), (20, 165, 16), (54, 15, 57), (12, 150, 9), (10, 46, 99), (94, 89, 46), (48, 37, 106),
                  (42, 10, 96), (7, 164, 128), (98, 213, 120), (40, 5, 219), (54, 25, 150), (251, 74, 172),
                  (0, 236, 196), (21, 104, 190), (226, 74, 232), (120, 67, 25), (191, 106, 197), (8, 15, 134),
                  (21, 2, 1), (142, 63, 109), (133, 148, 146), (187, 77, 253), (155, 22, 122), (218, 130, 77),
                  (164, 102, 79), (43, 152, 125), (185, 124, 151), (95, 159, 238), (128, 89, 85), (228, 6, 60),
                  (6, 41, 210), (11, 1, 133), (30, 96, 58), (230, 136, 109), (126, 45, 174), (164, 63, 165),
                  (32, 111, 29), (232, 40, 70), (55, 31, 198), (148, 211, 129), (10, 186, 211), (181, 201, 94),
                  (55, 35, 92), (129, 140, 233), (70, 250, 116), (61, 209, 152), (216, 21, 138), (100, 0, 176),
                  (3, 42, 70), (151, 13, 44), (216, 102, 88), (125, 216, 93), (171, 236, 47), (253, 127, 103),
                  (205, 137, 244), (193, 137, 224), (36, 152, 214), (17, 50, 238), (154, 165, 67), (114, 129, 60),
                  (119, 24, 48), (73, 8, 110)]
    if labels is None:
        # Define COCO Labels
        labels = {0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane',
                  6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant',
                  12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog',
                  18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe',
                  25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee',
                  31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat',
                  36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle',
                  41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana',
                  48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog',
                  54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed',
                  61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote',
                  67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink',
                  73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear',
                  79: u'hair drier', 80: u'toothbrush'}

    # plot each boxes
    for box in Boxes:
        # add score in label if score=True
        if score:
            label = labels[int(box[-1]) + 1] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1]) + 1]
        # filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(Image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(Image, box, label, color)

    # show image
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'Image/anchor_image_{kkk + 1}.jpg', Image)

    # 使用 matplotlib 显示图像
    # plt.imshow(Image)
    # plt.axis('off')
    # plt.show()


def draw_mask(Image, mask_generated):
    """
    
    :param Image: 
    :param mask_generated: 
    :return: 
    """
    masked_image = Image.copy()

    masked_image = np.where(mask_generated.astype(int),
                            np.array([0, 255, 0], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(Image, 0.3, masked_image, 0.7, 0)


def draw_masks_fromList(Image, Masks_generated, labels, colors):
    masked_image = Image.copy()
    for p in range(len(Masks_generated)):
        masked_image = np.where(np.repeat(Masks_generated[p][:, :, np.newaxis], 3, axis=2),
                                np.asarray(colors[int(labels[p][-1])], dtype='uint8'),
                                masked_image)

        masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(Image, 0.3, masked_image, 0.7, 0)


if __name__ == '__main__':

    # MODEL_TYPE：要使用的 SAM 架构
    # CHECKPOINT_PATH：包含模型权重的文件的路径
    # DEVICE：使用的处理器，“cpu”或“cuda”（如果 GPU 可用）
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    DEVICE = "cuda"  # cpu,cuda

    # 模型加载

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    dir_path = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data"

    all_image = os.listdir(dir_path)

    for i in range(len(all_image)):
        image_path = os.path.join(dir_path, all_image[i])
        # print(image_path)

        # 读取图像
        image = cv2.imread(image_path)
        # Display the shape of the array
        # print("Image shape:", image.shape)
        # Display the type of the array
        # print("Image type:", image.dtype)

        # 如果需要调整图像大小，可以使用 cv2.resize
        # image = cv2.resize(image, (new_width, new_height))

        # 绘制原始图像
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.title(f"Image {i + 1}")
        # plt.show()

        # 默认的segment anything
        # mask_generator = SamAutomaticMaskGenerator(sam)
        # masks_generated = mask_generator.generate(image)
        # print(masks_generated)
        # print(len(masks_generated))

        # area：遮罩区域（以像素为单位）
        # bbox：XYWH
        # 格式的掩模边界框
        # Predicted_iou：模型预测的掩模质量得分
        # point_coords：生成此掩码的采样输入点
        # stable_score：额外的掩模质量分数
        # Crop_box：用于生成XYWH格式的此蒙版的图像裁剪

        # 绘制遮罩图
        # plt.imshow(masks_generated[0]['segmentation'], cmap='gray')
        # plt.axis('off')
        # plt.show()

        # segmented_image = draw_masks_fromDict(image, masks_generated)
        # 保存图像到本地
        # cv2.imwrite('segmented_image.jpg',segmented_image)

        # plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        model = YOLO("yolov8n.pt")

        results = model.predict(image)

        print(results[0].to('cpu').boxes.data)
        if (len(results[0].to('cpu').boxes.data)) == 0:
            continue
        image_bboxes = image.copy()

        boxes = np.array(results[0].to('cpu').boxes.data)

        # 绘制锚框图
        plot_bboxes(image_bboxes, boxes, score=False, kkk=i)

        # 预测mask
        mask_predictor = SamPredictor(sam)
        mask_predictor.set_image(image)

        # 单个对象检测
        # mask, _, _ = mask_predictor.predict(
        #     box=boxes[1][:-2]
        # )
        # mask = np.transpose(mask, (1, 2, 0))
        #
        # segmented_image = draw_mask(image, mask)
        #
        # plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # 多个对象检测
        input_boxes = torch.tensor(boxes[:, :-2], device=mask_predictor.device)

        transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = mask_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = torch.squeeze(masks, 1)
        COLORS = [(89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98),
                  (130, 172, 179), (115, 209, 128), (204, 79, 135), (136, 126, 185), (209, 213, 45), (44, 52, 10),
                  (101, 158, 121), (179, 124, 12), (25, 33, 189), (45, 115, 11), (73, 197, 184), (62, 225, 221),
                  (32, 46, 52), (20, 165, 16), (54, 15, 57), (12, 150, 9), (10, 46, 99), (94, 89, 46), (48, 37, 106),
                  (42, 10, 96), (7, 164, 128), (98, 213, 120), (40, 5, 219), (54, 25, 150), (251, 74, 172),
                  (0, 236, 196), (21, 104, 190), (226, 74, 232), (120, 67, 25), (191, 106, 197), (8, 15, 134),
                  (21, 2, 1), (142, 63, 109), (133, 148, 146), (187, 77, 253), (155, 22, 122), (218, 130, 77),
                  (164, 102, 79), (43, 152, 125), (185, 124, 151), (95, 159, 238), (128, 89, 85), (228, 6, 60),
                  (6, 41, 210), (11, 1, 133), (30, 96, 58), (230, 136, 109), (126, 45, 174), (164, 63, 165),
                  (32, 111, 29), (232, 40, 70), (55, 31, 198), (148, 211, 129), (10, 186, 211), (181, 201, 94),
                  (55, 35, 92), (129, 140, 233), (70, 250, 116), (61, 209, 152), (216, 21, 138), (100, 0, 176),
                  (3, 42, 70), (151, 13, 44), (216, 102, 88), (125, 216, 93), (171, 236, 47), (253, 127, 103),
                  (205, 137, 244), (193, 137, 224), (36, 152, 214), (17, 50, 238), (154, 165, 67), (114, 129, 60),
                  (119, 24, 48), (73, 8, 110)]
        segmented_image = draw_masks_fromList(image, masks.to('cpu'), boxes, COLORS)

        cv2.imwrite(f'Image/segmented_image_{i + 1}.jpg', segmented_image)
        # plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        # 保存图像
