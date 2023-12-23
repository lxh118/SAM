# SAM
CV:SAM+YOLOv8+Medical Image Segmentation


终端执行下载模型：curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

python版本：Python 3.9.18


ground_truth_dir : mask的文件路径

image_dirPath : 原始图路径

###eg:
ground_truth_dir = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_GroundTruth"

image_dirPath = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data"


bbox_coords ：锚框字典

ground_truth_masks ：遮罩array

##eg：
bbox_coords = {}  # {'ISIC_0000000': array([ 51,  47, 899, 635]), }  # yolo计算的结果

ground_truth_masks = {}  # {'ISIC_0000000':array([False,Ture]),}
