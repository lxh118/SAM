# _*_ coding : utf-8 _*_
# @Time : 2023/12/23 12:00
# @Author : 娄星华
# @File : lib
# @Project : SAM


SAM_MODEL = 'sam_vit_h_4b8939.pth'
MODEL_TYPE = 'vit_h'

# curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# model_type = 'vit_b'
# checkpoint = 'sam_vit_b_01ec64.pth'
# model_type = 'vit_h'
# checkpoint = 'sam_vit_h_4b8939.pth'

DEVICE = 'cuda'

ground_truth_dir = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_GroundTruth"
image_dirPath = r"D:\MyFile\DataSet\ISBI2016_ISIC_Part1\ISBI2016_ISIC_Part1_Training_Data"

EPOCHS = 20
