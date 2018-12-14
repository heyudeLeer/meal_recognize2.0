#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from PIL import ImageEnhance
    from PIL import ImageStat
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
import numpy as np

def enhance_by_random(data_info=None,x=None,color_range=None,brightness_range=None,contrast_range=None,sharpness_range=None):

    # x = array_to_img(x)
    index = data_info.enhance_index
    name = []

    if index == 0 or index > 3 :
        index += 1
        if len(brightness_range) != 2:
            raise ValueError('`brightness_range should be tuple or list of two floats. '
                             'Received arg: ', brightness_range)
        imgenhancer_Brightness = ImageEnhance.Brightness(x)
        u = np.random.uniform(brightness_range[0], brightness_range[1])
        x = imgenhancer_Brightness.enhance(u)
        name.append('_brightness'+str(u))

    if index == 1 or index > 3:
        index += 1
        if len(color_range) != 2:
            raise ValueError('`color_range should be tuple or list of two floats. '
                             'Received arg: ', color_range)
        imgenhancer_Color = ImageEnhance.Color(x)
        u = np.random.uniform(color_range[0], color_range[1])
        x = imgenhancer_Color.enhance(u)
        name.append('_color'+str(u))

    if index == 2 or index > 3 :
        index += 1
        if len(contrast_range) != 2:
            raise ValueError('`contrast_range should be tuple or list of two floats. '
                             'Received arg: ', contrast_range)
        imgenhancer_Contrast = ImageEnhance.Contrast(x)
        u = np.random.uniform(contrast_range[0], contrast_range[1])
        x = imgenhancer_Contrast.enhance(u)
        name.append('_contrast'+str(u))

    if index == 3 or index > 3:
        index += 1
        if len(sharpness_range) != 2:
            raise ValueError('`sharpness_range should be tuple or list of two floats. '
                             'Received arg: ', sharpness_range)
        imgenhancer_Sharpness = ImageEnhance.Sharpness(x)
        u = np.random.uniform(sharpness_range[0], sharpness_range[1])
        x = imgenhancer_Sharpness.enhance(u)
        name.append('_sharp'+str(u))

    #x = img_to_array(x)
    img_enhance_Contrast.save(name+".jpg")
    if index == 12:
        index = 0

    data_info.enhance_index = index
    return x


if __name__ == '__main__':

    img_url1 = '/home/heyude/data_set/model/predictImg/img/fdbd7eda-14e3-4aa4-88f5-1dd4252a6c2f.jpg'
    img_url2 = '/home/heyude/data_set/model/predictImg/img/78288410-6ecd-4fed-ac51-0e472a078ce0.jpg'

    img = pil_image.open(img_url1)

    ##图像处理##

    # 转换为RGB图像
    img = img.convert("RGB")
    imgbri = img.point(lambda i: i * 1.4)  # 对每一个像素点进行增强
    imgbri.save("1bri.jpg")
    imgbri.show()

    # PIL图像增强ImageEnhance
    istep = 4
    irange = 4.0

    imgenhancer_Color = ImageEnhance.Color(img)
    for i in range(istep):
        factor = i / irange + 1
        img_enhance_color = imgenhancer_Color.enhance(factor)
        img_enhance_color.show("Color %f" % factor)
        img_enhance_color.save("Color_%.2f.jpg" % factor)

    imgenhancer_Brightness = ImageEnhance.Brightness(img)
    for i in range(istep):
        factor = i / irange +1
        img_enhance_Brightness = imgenhancer_Brightness.enhance(factor)
        img_enhance_Brightness.show("Brightness %f" % factor)
        img_enhance_Brightness.save("Brightness_%.2f.jpg" % factor)

    imgenhancer_Contrast = ImageEnhance.Contrast(img)
    for i in range(istep):
        factor = i / irange + 1
        img_enhance_Contrast = imgenhancer_Contrast.enhance(factor)
        img_enhance_Contrast.show("Contrast %f" % factor)
        img_enhance_Contrast.save("Contrast_%.2f.jpg" % factor)

    imgenhancer_Sharpness = ImageEnhance.Sharpness(img)
    for i in range(istep):
        factor = i / irange + 1
        img_enhance_Sharpness = imgenhancer_Sharpness.enhance(factor)
        img_enhance_Sharpness.show("Sharpness %f" % factor)
        img_enhance_Sharpness.save("Sharpness_%.2f.jpg" % factor)
        # end