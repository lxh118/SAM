import lib
import os


def Rename_CT():
    raw_path = lib.image_dirPath

    # 设定你要重命名文件的目录
    directory = raw_path
    # 列出目录下所有文件
    files = os.listdir(directory)
    # 通过循环对每个文件执行操作
    for filename in files:
        # 创建新的文件名（这里你需要根据自己的需求来自定义重命名逻辑）

        new_filename = filename.replace("tif", "png")

        # 完整的文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # 重命名操作
        os.rename(old_file, new_file)


def Rename_ISBI2016():
    folder_path = lib.ground_truth_dir  # 替换成你的文件夹路径

    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_Segmentation.png"):
            # 构建新的文件名并重命名
            new_filename = file_name.replace("_Segmentation", "")
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} to {new_filename}")


if __name__ == "__main__":
    # Rename_ISBI2016()
    Rename_CT()
    pass
