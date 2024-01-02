# _*_ coding : utf-8 _*_
# @Time : 2023/12/29 18:02
# @Author : 娄星华
# @File : DrawImage
# @Project : SAM
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean


def Epoch_loss(Train_losses, Val_losses):
    train_mean_losses = [mean(X) for X in Train_losses]
    val_mean_losses = [mean(X) for X in Val_losses]
    plt.plot(list(range(len(train_mean_losses))), train_mean_losses)
    plt.plot(list(range(len(val_mean_losses))), val_mean_losses)
    plt.title('Mean epoch loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig("Image/loss.png")
    plt.show()


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
    pass
