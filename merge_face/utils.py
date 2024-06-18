import cv2
import numpy as np


# ====================
# 提取图像的Alpha通道
# param image: 图像，四通道
# return: 二值图像（黑白图像）
# ====================
def convert_png_to_black_and_white(image):
    # 提取图像的Alpha通道
    alpha_channel = image[:, :, 3]

    # 将Alpha通道转换为二值图像（黑白图像）
    _, binary_image = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # 创建一个与原始图像大小相同的白色图像
    white_image = 255 * np.ones_like(image[:, :, :3])

    # 将Alpha通道为0的像素设为黑色
    white_image[binary_image == 0] = [0, 0, 0]

    return white_image


# 放大/缩小图像
def resize_png(image, scale=0.9):
    # 缩小图像尺寸
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)

    # 创建与原始图像大小相同的黑色背景
    if scale < 1:
        background = np.zeros_like(image)
        # 计算在背景上粘贴图像的起始位置
        start_x = int((background.shape[1] - resized_image.shape[1]) / 2)
        start_y = int((background.shape[0] - resized_image.shape[0]) / 2)
    else:
        background = np.ones_like(resized_image)
        # 计算在背景上粘贴图像的起始位置
        start_x = int((resized_image.shape[1] - background.shape[1]) / 2)
        start_y = int((resized_image.shape[0] - background.shape[0]) / 2)

    # 将缩小后的图像复制到背景上
    background[start_y:start_y + resized_image.shape[0], start_x:start_x + resized_image.shape[1]] = resized_image

    return background


def to_blurry(face, mask, bg, ksize=35):
    ksize = (ksize, ksize)
    mask = cv2.GaussianBlur(mask, ksize, 0)

    mask = mask / 255  # 除以 255，計算每個像素的黑白色彩在 255 中所佔的比例

    img = face
    img = img / 255  # 除以 255，計算每個像素的色彩在 255 中所佔的比例

    bg = bg / 255  # 除以 255，計算每個像素的色彩在 255 中所佔的比例

    out = bg * (1 - mask) + img * mask  # 根據比例混合
    out = (out * 255).astype('uint8')  # 乘以 255 之後轉換成整數

    return out




def draw_img_with_mask(img: np.ndarray, mask: np.ndarray, bg: np.ndarray, ksize: int=0, top: int=0, left: int=0) -> np.ndarray:
    """
    把img和mask融合到bg上
    args:
        img: 原始图片, np.ndarray的三通道
        mask: 人脸的蒙版, np.ndarray的单通道
        bg: 背景图, np.ndarray的三通道
        ksize: 高斯模糊的ksize, 默认为0, 不进行高斯模糊
    return:
        result: 融合后的图像, np.ndarray的三通道
    """
    assert len(img.shape) == 3, "img必须是三通道的"
    assert len(bg.shape) == 3, "bg必须是三通道的"
    assert len(mask.shape) == 2, "mask必须是单通道的灰度图"
    height, width, _ = img.shape
    mask_h, mask_w = mask.shape
    assert height == mask_h and width == mask_w, "img和mask的尺寸必须一致"
    if ksize != 0:
        assert ksize % 2 == 1, "ksize必须为奇数"
        blur_ksize = (ksize, ksize)
        mask = cv2.GaussianBlur(mask, blur_ksize, 0)
    # 规范化图像
    img = img / 255
    mask = mask / 255
    bg = bg / 255
    # 蒙版应用于原图
    source = img * mask[:, :, np.newaxis]
    # 蒙版取反, 用于背景图
    mask_inv = 1 - mask
    bg[top:top+height, left:left+width] = bg[top:top+height, left:left+width] * mask_inv[:, :, np.newaxis]
    # 背景图和原图相加
    bg[top:top+height, left:left+width] = bg[top:top+height, left:left+width] + source
    result = (bg * 255).astype(np.uint8)  # 必须是cv_8u, 否则无法保存, imshow可以显示
    return result

def create_mask(width: int, height: int, gap: int=20):
    image = np.zeros((height, width), np.uint8)
    image[:, :] = 0  # 黑色背景
    image[gap:height - gap, gap:width - gap] = 255  # 中间白色方块
    return image

