from frcnn import FRCNN
import numpy as np
from PIL import Image
from utils.utils import (cvtColor, get_new_img_size, resize_image)

frcnn = FRCNN()

while True:
    img = input("Input image filename:")
    try:
        image = Image.open(img)
    except:
        print("Open Error! Try again!")
        continue
    else:
        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 计算resize后的图片的大小，resize后的图片短边为600
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        # 按照得到的尺寸对原图像进行resize
        r_image = resize_image(image, [input_shape[1], input_shape[0]])
        r_image.show()