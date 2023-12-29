# SAM
CV:SAM+YOLOv5+Medical Image Segmentation


终端执行下载模型：`curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

python版本：Python 3.9.18

---

ground_truth_dir : mask的文件路径

image_dirPath : 原始图路径

### eg:
ground_truth_dir = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_GroundTruth"

image_dirPath = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data"

---

bbox_coords ：锚框字典

ground_truth_masks ：遮罩array

### eg：
bbox_coords = {}  # {'ISIC_0000000': array([ 51,  47, 899, 635]), }  # yolo计算的结果

ground_truth_masks = {}  # {'ISIC_0000000':array([False,Ture]),}

---

# YOLO
yolov5代码使用：

1. clone ultralytics的yolov5代码

`git clone git@github.com:ultralytics/yolov5.git`

2. 使用labelimg制作yolo数据集，编辑CT_lung.yaml文件
3. 训练

`python ./train.py --data ./data/CT_lung.yaml --epochs 100 --batch-size 16 --imgsz 512 --optimizer 'Adam'`

4. 测试

`python detect.py --weights ./runs/train/exp/weights/last.pt --source ./datasets/CT/test --data ./data/CT_lung.yaml --imgsz 512 `

# 数据集下载地址
https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
https://challenge.isic-archive.com/data/
