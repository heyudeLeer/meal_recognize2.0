
# encoding: utf-8
print 'in predic'

import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import uniout
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import cv2
import warnings
import datetime

import data_label
import train
from keras.layers import Input, Activation
from keras.models import Model
from open_cvision import opencv_tools as cv_tools

#zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

class JudgeInfo:
    def __init__(self):

        self.object_min_rate = 0.2  # area rate
        self.pixel_min_num = 8
        #self.degreeCorrectTh = 0.2  # fullness

#RGB item for Image segmentation
def targetColorInit(class_num=0):

    if class_num > 0xFFFFFF:
        print ' the categorical num must be smaller than 0xFFFFFF in color segment'
        exit(0)

    targetColor = []
    colorAvg = (0xFF-0x20) / (class_num)

    for i in range(1, class_num+1):
    #for i in range(0, class_num ): #屏蔽bg

        color = colorAvg

        #r = (color >> 16) & 0xFF
        #g = (color >> 8) & 0xFF
        #b = color & 0xFF

        r = (0x20 + color*i) & 0xff
        g = (0x20 + color*i*2) & 0xff
        b = (0x20 + color*i*4) & 0xff

        targetColor.append((r, g, b)) #
        # print targetColor[i]

    return targetColor

def sigmoid2softmax_3D(array=None):

    x,y,z = array.shape

    # This returns a tensor
    inputs = Input(shape=(z,))
    # a layer instance is callable on a tensor, and returns a tensor
    predictions = Activation('softmax', name='block3_sepconv2_act')(inputs)
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)

    for i in range(x):
        for j in range(y):
            x = np.expand_dims(array[i,j,:], axis=0)
            pred= model.predict(x=x)
            array[i, j, :] = pred[0]

    del model
    return array


def getRgbImgFromUpsampling(imgP=None, data_info=None, check=False,coordinate=False):
    '''
    Summarize the segModel results
    :param imgP: segModel predict
    :return:rgbImg indicate the categories,location; dishes_dictory:(category pixels, name, coordinate(xl, yl, xr, yr))
    '''
    judgeInfo = JudgeInfo()

    mylist=[]
    targetColor = targetColorInit(class_num =data_info.class_num)
    rgbImg = np.zeros((data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, 3), dtype='uint8')

    avg_p = 1.0 / data_info.class_num

    #imgP[0] = sigmoid2softmax_3D(array=imgP[0])

    #统计可能有的object，存于myset
    #标准：max_1st > 0.8 and max_2nd < 0.2
    #过严，可能遗失类别,过松没有过滤作用，对后面的解析造成干扰。由于物体有局部特征相似的点，所以可能多了
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            a = imgP[0, i, j, :].copy()
            max_1st=np.max(a)
            max_index = a.argmax()
            a[max_index]=0
            max_2nd=np.max(a)
            if max_1st > 0.9  and max_2nd < avg_p/2 :
                rgbIndex = max_index

                mylist.append(rgbIndex) #保存class index

    myset = set(mylist)  # myset是另外一个列表，里面的内容是mylist里面的无重复项
    #print myset
    #删除势力弱的object
    for index in myset.copy():
        dot = mylist.count(index)  # value is dishes dot nums
        name = data_info.class_name_dic_t.get(index)  # item is dish index; key is dish name
        #print name
        #print dot
        if dot < judgeInfo.pixel_min_num:   #remove pixel过小的
            myset.remove(index)
            print 'Rgb remove '+name
        #else:
        #    print index, dot
    #print myset

    #将可能的object，用rgb标示出来
    #标准：median < 0.2,object > 0.8且唯一
    '''
    object_pixel_sum = 0
    mylist = []
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            rgbIndex = 0
            object_candidate = {}
            # 此处需要把softmax, imgP[0, i, j, :] 转化到(0,1) add to 1;    #########  !!!!!!!!!!!!!!!!!!!
            if data_info.class_num == 2:
                median_x = min(imgP[0, i, j, :])
            else:
                median_x = np.median(imgP[0, i, j, :])
            if median_x >= 0.3:
                continue
            object_pixel_sum += 1
            for index in myset:
                object = imgP[0, i, j, index]
                if  object > 0.7:
                    rgbIndex = index
                    name = data_info.class_name_dic_t[index]
                    object_candidate[name] = object

            leng = len(object_candidate)
            if leng == 1 :
                rgbImg[i, j, :] = targetColor[rgbIndex]  # 给像素赋值，以示区别
                mylist.append(rgbIndex)  # 保存class index
            elif leng== 0:
                #print i,j
                #print
                rgbImg[i, j, :] = (128,128,128)
            else:
                #print i, j
                #print object_candidate
                #print
                rgbImg[i, j, :] = (255, 255, 255)
    '''
    object_pixel_sum = 0
    mylist = []
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            a = imgP[0, i, j, :]
            if data_info.class_num == 2:
                median_x = min(a)
            else:
                median_x = np.median(a)

            if median_x < avg_p/2:
                object_pixel_sum += 1

                max_index = a.argmax()
                object = imgP[0, i, j, max_index]
                if max_index in myset and object > 0.6:
                    mylist.append(max_index)  # 保存class index
                    rgbImg[i, j, :] = targetColor[max_index]  # 给像素赋值，以示区别
                else:
                    rgbImg[i, j, :] = (128,128,128)


    dishes_dictory = {}
    for item in myset:
        name = data_info.class_name_dic_t.get(item)  # item is dish index; key is dish name
        dot = mylist.count(item)  # value is dishes dot nums
        object_num = dot * 1.125 / data_info.object_pixels_avg[item] # 1.125 泛化修正因子

        #object_num = round(object_num)
        if object_num > judgeInfo.object_min_rate :
            #dishes_dictory[name] = np.uint32(object_num)
            dishes_dictory[name] = object_num

    '''
    # get coordinate
    xl, yl, xr, yr = 0, 0, 0, 0
    if coordinate:
        i_list = []
        j_list = []
        for i in range(data_info.IMG_ROW_OUT):
            for j in range(data_info.IMG_COL_OUT):
                if rgbImg[i, j, 0] == targetColor[item][0] and \
                                rgbImg[i, j, 1] == targetColor[item][1] and \
                                rgbImg[i, j, 2] == targetColor[item][2]:
                    i_list.append(i)
                    j_list.append(j)
        xl = min(j_list)
        yl = min(i_list)
        xr = max(j_list)
        yr = max(i_list)

    #dishes_dictory[key] = (dot, name, (xl, yl, xr, yr))
    '''

    return rgbImg, dishes_dictory


def get_rgb_mark(imgP=None, data_info=None):

    targetColor = targetColorInit(class_num =data_info.class_num)
    rgbImg = np.zeros((data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, 3), dtype='uint8')

    avg_p = 1.0/data_info.class_num
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            a = imgP[0, i, j]
            max_index = a.argmax()
            object = imgP[0, i, j, max_index]
            if object > avg_p*2:

                if object >= data_info.confidence_threshold: # 0.9
                    rgbImg[i, j, :] = targetColor[max_index]  # 给像素赋值，以示区别
                elif object >=0.6:
                    rgbImg[i, j, :] = (144,144,144)
                elif object >=0.3:
                    rgbImg[i, j, :] = (96,96,96)
                else:
                    rgbImg[i, j, :] = (48,48,48)

    return rgbImg


def get_contours(img=None,th=0):

    ret, thresh = cv2.threshold(src=img,thresh=th,maxval=255, type=cv2.THRESH_BINARY)
    if cv2.__version__[0] == str(2):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL RETR_LIST
    else:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_confidence(img=None,th=0):
    ret, thresh = cv2.threshold(src=img, thresh=th, maxval=0,type=cv2.THRESH_TOZERO)
    #x = thresh
    #print(type(x), x.shape, x.dtype, np.min(x), np.max(x))
    thresh = thresh.astype(np.float)
    #x = thresh
    #print(type(x), x.shape, x.dtype, np.min(x), np.max(x))

    #print thresh[20:30,24]
    thresh[thresh==0.0] = np.nan
    #print thresh[20:30,24]
    means = np.nanmean(thresh)/255.
    #print means
    return means


def get_dishes_with_confidence(dataInfo=None,segImg=None):

    dishes_dictory = {}
    img = np.uint8(segImg * 255)

    for index in range(dataInfo.class_num):
        contours = get_contours(img=img[:,:,index], th=dataInfo.threshold_value)
        name = dataInfo.class_name_dic_t.get(index)
        dish_num = 0
        for contour in contours:
            #contour = cv2.convexHull(contour)
            area = cv2.contourArea(contour)
            object_rate = area / dataInfo.object_area_avg.get(name)

            if object_rate > 1.75:
                warnings.warn('dishes maybe linked...')

            if object_rate > 0.25:  # 全局因子，像素超过采样均值0.25即认为是独立的菜
                integer = np.uint8(object_rate)
                mod = object_rate - integer
                if integer == 0:
                    dish_num += 1
                else:
                    dish_num += integer
                    if mod > 0.75:  # 浮点舍去的若超过采样均值的0.75则认为是另一盘菜，等待用神经网络再次识别，待提升
                        dish_num += 1

        if dish_num > 0:
            dishes_dictory[name] = dish_num

    return dishes_dictory


def get_dishes_with_confidence_debug(dataInfo=None,segImg=None):

    dishes = []
    img = np.uint8(segImg * 255)

    x, y, z = segImg.shape
    seg_contour = np.zeros((x, y, z), dtype=np.uint8)

    for index in range(dataInfo.class_num):
        contours = get_contours(img=img[:, :, index], th=dataInfo.threshold_value)
        name = dataInfo.class_name_dic_t.get(index)
        dish_num = 0
        dish_contour = 0

        area_fraction = []
        canvas = np.zeros((x, y), dtype=np.uint8)

        for contour in contours:
            #contour = cv2.convexHull(contour)
            area = cv2.contourArea(contour)
            object_rate = area / dataInfo.object_area_avg.get(name)
            area_fraction.append(object_rate)
            cv2.drawContours(image=canvas, contours=[contour], contourIdx=-1, color=255, thickness=1)
            dish_contour += 1

            if object_rate > 1.75:
                warnings.warn('dishes maybe linked...')

            if object_rate > 0.25:  # 全局因子，像素超过采样均值0.25即认为是独立的菜
                integer = np.uint8(object_rate)
                mod = object_rate - integer
                if integer == 0:
                    dish_num += 1
                else:
                    dish_num += integer
                    if mod > 0.75:  # 浮点舍去的若超过采样均值的0.75则认为是另一盘菜，等待用神经网络再次识别，待提升
                        dish_num += 1

            confidence = get_confidence(img=img[:,:,index], th=dataInfo.threshold_value)
            confidence = round(confidence,5)

        seg_contour[:,:,index]  = canvas

        if dish_contour > 0:
            dishes_dictory = {}
            dishes_dictory['name'] = name
            dishes_dictory['num'] = dish_num
            dishes_dictory['confidence'] = confidence
            dishes_dictory['area_fraction'] = area_fraction
            dishes.append(dishes_dictory)
    print dishes
    return dishes,seg_contour


# predict
def getDishesBySegImg(dataInfo=None,segImg=None,drawCnt=0):
    '''
    :param dataInfo:
    :param segImg:
    :param drawCnt: 0:no draw cnt; 1:1by1; 2:all cnt in one canvas
    :return:
    '''

    judgeInfo = JudgeInfo()
    dishes_dictory = {}

    canvas = None
    canvas_l = []
    x, y, _ = segImg.shape
    if drawCnt > 0:
        canvas = np.zeros((x, y), dtype=np.uint8)

    for item in range(dataInfo.class_num):

        areas,canvas = cv_tools.getAreaOneDimension(img=segImg[:,:,item], th=dataInfo.threshold_value,canvas=canvas)
        if areas is None:
            if drawCnt == 1:
                canvas_l.append(canvas)
                canvas = np.zeros((x, y), dtype=np.uint8)
            continue

        object_num = 0
        name = dataInfo.class_name_dic_t.get(item)  # item is dish index; key is dish name

        #print
        #print name +'_avg: '+ str(dataInfo.object_area_avg[item])
        #print areas

        for area in areas:
            object_rate = area / dataInfo.object_area_avg[item]

            if object_rate > judgeInfo.object_min_rate:
                object_num += 1
                #print 'get int '+ str(area) + ':' + str(object_rate)

        if object_num == 1 and object_rate > 1.0:
            object_num = round(object_rate)
            #print 'get round ' + str(object_rate) +":"+str(object_num)

        if object_num > 0:
            dishes_dictory[name] = object_num

        if drawCnt == 1:
            canvas_l.append(canvas)
            canvas = np.zeros((x, y), dtype=np.uint8)

    if drawCnt == 1:
        return dishes_dictory,canvas_l
    else:
        return dishes_dictory,canvas


def imgPredict2DShow(url=None,predict_info=None):
    '''
    show the 2D output for predict Visualization
    :param url: image path
    :return: plt show result
    '''
    title = 'fea_map'
    img = data_label.loadImage(url=url,data_info=predict_info)

    y_p, segmentation_map = predict_info.model.predict(img)

    n=predict_info.class_num/2
    plt.figure(figsize=(20, 5))
    for i in range(n):

        ax = plt.subplot(2, n, i+1)
        ax.set_title(title+str(i))
        ax.imshow(segmentation_map[0, :, :, i])
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i+1 + n)
        ax.set_title(title+str(i+n))
        ax.imshow(segmentation_map[0, :, :, i+n])
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def imgPredict2DShow_diff(url=None,predict_info=None):
    '''
    show the 2D output for predict Visualization
    :param url: image path
    :return: plt show result
    '''
    title = 'fea_map'
    img1 = data_label.loadImage(url=url[0],data_info=predict_info)
    img2 = data_label.loadImage(url=url[1], data_info=predict_info)

    y_p1, segmentation_map1 = predict_info.model.predict(img1)
    segImg1, dishes_info = getRgbImgFromUpsampling(imgP=segmentation_map1, data_info=predict_info)
    y_p2, segmentation_map2 = predict_info.model.predict(img2)
    segImg2, dishes_info = getRgbImgFromUpsampling(imgP=segmentation_map2, data_info=predict_info)


    print predict_info.class_num
    n=np.uint8(predict_info.class_num) + 2
    plt.figure(figsize=(20, 10))

    for i in range(n-2):

        ax = plt.subplot(2, n, i+1)
        seg_mean1 = round(np.mean(segmentation_map1[0, :, :, i]),3)
        ax.set_title(predict_info.class_name_dic_t[i][0:6] + str(seg_mean1))
        ax.imshow(segmentation_map1[0, :, :, i])
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i+1+n )
        seg_mean2 = round(np.mean(segmentation_map2[0, :, :, i]),3)
        ax.set_title(str(seg_mean2)+predict_info.class_name_dic_t[i][0:6])
        ax.imshow(segmentation_map2[0, :, :, i])
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n-1)
    ax.imshow(segImg1)
    ax = plt.subplot(2, n, n)
    ax.imshow(img1[0] / 255.)

    ax = plt.subplot(2, n, n+n-1)
    ax.imshow(segImg2)
    ax = plt.subplot(2, n, n+n)
    ax.imshow(img2[0] / 255.)


    plt.show()


def segImgfile_web(data_info= None, url=None,out_path=None, show=False):

    print ('predict  ' + url)

    img = data_label.loadImage(url=url,data_info=data_info) #113ms
    y_p, pred = data_info.model.predict(img)  #70ms

    if show is True:
        RgbImg = get_rgb_mark(imgP=pred, data_info=data_info)
        dishes_info, seg_contour = get_dishes_with_confidence_debug(dataInfo=data_info, segImg=pred[0])
    else:
        dishes_info = get_dishes_with_confidence(dataInfo=data_info, segImg=pred[0])

    if show is True:

        n = 3
        m  = data_info.class_num + 1
        plt.figure(figsize=(20, 8))
        # display source img
        ax = plt.subplot(n, m, 1)
        ax.imshow(img[0]/255.)
        ax.set_title(os.path.basename(url))
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

        ax = plt.subplot(n, m, 2)
        # display result
        ax.imshow(RgbImg)
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

        for i in range(data_info.class_num):
            ax = plt.subplot(n, m, m+1 + i)
            ax.set_title(data_info.class_name_dic_t.get(i))
            ax.get_xaxis().set_visible(False)
            ax.imshow(seg_contour[:,:,i])

        for i in range(data_info.class_num - 1):
            seg_sum = np.sum(seg_contour,axis=2)
        ax = plt.subplot(n, m, 3)
        ax.imshow(seg_sum)
        ax.get_xaxis().set_visible(False)

        for i in range(data_info.class_num):
            ax = plt.subplot(n, m, m*2+1 + i)
            ax.get_xaxis().set_visible(False)
            ax.imshow(pred[0,:,:,i])

        pixel_name = os.path.basename(url)
        #pixel_name = pixel_name[0:-4]
        (shotname, extension) = os.path.splitext(pixel_name)
        pixel_name = shotname
        path = out_path + '/predictImg/'
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        plt.savefig( path + pixel_name +'_pcla.jpg')

        plt.show()
        plt.close()

    #dishes_info = sorted(dishes_info.items(), key=lambda e: e[0], reverse=False)

    return dishes_info


def segImgDir(PredictInfo=None, segPath=None):
    '''
    images in folder segmentation
    :param segPath: images path
    :return: print and plt show result
    '''
    i=0
    judgeInfo = JudgeInfo()
    print segPath
    for _, _, files in os.walk(segPath):
        print ("coming " + segPath)
        break
    n = len(files)
    print 'files num is ' + str(n)

    plt.figure(figsize=(20, 5))
    for file in files:

        if cmp(file, "category.txt") == 0:
            continue

        print ('predict  '+file)
        img = data_label.loadImage(url=segPath + '/' + file,data_info=PredictInfo)
        y_p, pred = PredictInfo.model.predict(img)
        #print pred[0].shape
        #print pred[1].shape
        segImg,dishes_info = getRgbImgFromUpsampling(imgP=pred, data_info=PredictInfo)

        i += 1
        # display source img
        ax = plt.subplot(2, n, i)
        ax.set_title(file[-8:-1])
        plt.imshow(img[0]/255.)

        # ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i+n)
        for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
            #xl = dish_v[2][0]
            #yl = dish_v[2][1]
            #xr = dish_v[2][2]
            #yr = dish_v[2][3]
            #h = xr - xl + 1
            #w = yr - yl + 1
            #area = h * w
            #areaRate = dish_v * 1.0 / (PredictInfo.IMG_ROW_OUT * PredictInfo.IMG_COL_OUT)
            #fullness = dish_v[0] * 1.0 / area
            #if areaRate > judgeInfo.tageCorrectTh: #and fullness > judgeInfo.degreeCorrectTh:
            print("the %s mostly has found,number of pixels and areaRate:%f" % (dish_k, dish_v))
                #central_x = (xr + xl) / 2
                #central_y = (yr + yl) / 2
                #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
                #ax.text(xl, central_y, dish_v[1].decode('utf8'), size=8, color="r", fontproperties=zhfont1)

        # display result
        ax.imshow(segImg)
        # ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def segImgDir_cantai(PredictInfo=None, segPath=None, plca_path=None):
    '''
    images in folder segmentation
    :param segPath: images path
    :return: print and plt show result
    '''
    i=0
    judgeInfo = JudgeInfo()
    print segPath
    for _, _, files in os.walk(segPath):
        print ("coming " + segPath)
        break
    n = len(files)/3
    print 'files num is ' + str(n)

    plt.figure(figsize=(20, 8))
    for file in files:

        if cmp(file, "category.txt") == 0:
            continue

        print
        print ('predict  '+file)
        img = data_label.loadImage(url=segPath + '/' + file,data_info=PredictInfo)
        y_p, pred = PredictInfo.model.predict(img)
        #print pred[0].shape
        #print pred[1].shape
        RgbImg,dishes_info = getRgbImgFromUpsampling(imgP=pred, data_info=PredictInfo)
        dishes_info, canvas = getDishesBySegImg(dataInfo=PredictInfo, segImg=pred[0],drawCnt=2)

        i += 1
        # display source img
        ax = plt.subplot(3, n, i)
        ax.set_title(file[-8:-1])
        plt.imshow(img[0]/255.)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+n)
        for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
            #xl = dish_v[2][0]
            #yl = dish_v[2][1]
            #xr = dish_v[2][2]
            #yr = dish_v[2][3]
            #h = xr - xl + 1
            #w = yr - yl + 1
            #area = h * w
            #areaRate = dish_v * 1.0 / (PredictInfo.IMG_ROW_OUT * PredictInfo.IMG_COL_OUT)
            #fullness = dish_v[0] * 1.0 / area
            #if areaRate > judgeInfo.tageCorrectTh: #and fullness > judgeInfo.degreeCorrectTh:
            print("the %s mostly has found,pixels area rate:%f" % (dish_k, dish_v))
                #central_x = (xr + xl) / 2
                #central_y = (yr + yl) / 2
                #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
                #ax.text(xl, central_y, dish_v[1].decode('utf8'), size=8, color="r", fontproperties=zhfont1)

        # display result
        ax.imshow(RgbImg)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        pixel_name = os.path.basename(file)
        # pixel_name = pixel_name[0:-4]
        (shotname, extension) = os.path.splitext(pixel_name)
        #img = data_label.loadImage(url=plca_path + '/' + shotname+'_pcla.jpg',data_info=PredictInfo)
        ax = plt.subplot(3, n, i+n*2)
        ax.set_title(shotname[-4:-1])
        plt.imshow(canvas/255.)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n:
            plt.show()
            cv2.waitKey()
            plt.figure(figsize=(20, 5))
            i = 0


def getPredPixels(imgP=None, data_info=None):

    avg_p = 1.0 / data_info.class_num * 0.5 #* 1e-4
    object_pixel_sum = 0

    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            a = imgP[0, i, j, :]
            if data_info.class_num == 2:
                median_x = min(a)
            else:
                median_x = np.median(a)

            max_index = a.argmax()
            object = imgP[0, i, j, max_index]

            if median_x < avg_p and object > 0.6:
                object_pixel_sum += 1

    return object_pixel_sum


# train

def save_pred_image(img=None, labels=None,data_info=None,out_path=None,titels=None,x_labels=None):
    '''
    show the 2D output for predict Visualization
    :param url: image path
    :return: plt show result   #plt 和 ax 搞混了，日后做实验
    '''
    img = img.copy()
    img = np.uint8(img)
    var_tabel = np.var(labels,axis=2)

    plt.figure(figsize=(20, 5))
    plt.suptitle(titels,fontsize=20)

    for i in range(data_info.class_num):
        ax = plt.subplot(1, data_info.class_num+2, i+1)
        ax.set_title(x_labels[i],fontsize=12)
        # ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        label = labels[:, :, i]
        #print(type(label), label.dtype, np.min(label), np.max(label))
        ax.imshow(label)
        #plt.subplots_adjust(wspace=1, hspace=1)  # 调整子图间距

    ax = plt.subplot(1, data_info.class_num + 2, data_info.class_num + 1)
    #ax.set_title(titels[1])
    ax.imshow(img)
    #plt.xlabel(titels[2])

    ax = plt.subplot(1, data_info.class_num + 2, data_info.class_num + 2)
    ax.set_title('var')
    ax.imshow(var_tabel)

    save_path = out_path + '/predictImg/trained_warnings'
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 现在
    plt.savefig(save_path + '/' + nowTime + '.jpg')
    plt.close()


def getPerPixels(setsPath=None, data_info=None,out_path=None):

    ret = 0
    objects = data_info.class_num
    for item in range(objects):

        dirName = data_info.class_name_dic_t.get(item)  # item is dish index; key is dish name
        if dirName == 'bg':  # 抑制背景，效果还不错，也够简单
            #data_info.object_pixels_avg.append(1.0)
            data_info.object_area_avg['bg'] = 1.0
            continue

        for _, _, files in os.walk(setsPath + '/' + dirName):
            break

        if len(files)==0:
            #data_info.object_pixels_avg.append(250)
            data_info.object_area_avg[dirName] = 250
            continue

        #dot_sum = 0
        area_sum = 0
        for name in files:
            if cmp(file, "category.txt") == 0:
                continue
            url = setsPath + '/' + dirName + '/' + name
            print ('predict ' + url)

            img = data_label.loadImage(url=url,data_info=data_info)
            y_p,preds = data_info.model.predict(img)

            x_labels = []
            y = np.zeros((1,data_info.class_num),dtype=np.uint8)
            y[0,item] = 1
            for index in range(data_info.class_num):
                x_labels.append(str(y[0, index]) + '/' + str(round(y_p[0, index], 3)))

            #dot = getPredPixels(imgP=preds, data_info=data_info)
            #dot_sum  += dot
            #print dot
            label = preds[0,:,:,item]
            areas,_ = cv_tools.getAreaOneDimension(img=label,th=data_info.threshold_value)
            if areas is None:
                warnings.warn('train failed!! predict a train_set image,but contour is None')
                save_pred_image(img=img[0], labels=preds[0], data_info=data_info,
                                out_path=out_path,titels='pred none:'+name, x_labels=x_labels)
                ret = -1
            else:
                area = max(areas)
                if len(areas) > 1:
                    warnings.warn('found more object,please check!')
                    save_pred_image(img=img[0], labels=preds[0], data_info=data_info,
                                    out_path=out_path,titels='more object:'+name, x_labels=x_labels)
                    ret = 1
                elif area > 2000 or area < 50:
                    warnings.warn('object so small or big,please check!')
                    save_pred_image(img=img[0], labels=preds[0], data_info=data_info,
                                    out_path=out_path,titels='too big_small:'+name, x_labels=x_labels)
                    ret = 1

                area_sum += area
                print area

        #pixel_avg = dot_sum * 1.0 / len(files)
        area_avg = area_sum * 1.0 / len(files)

        #pixel_avg = round(pixel_avg,2)
        area_avg = round(area_avg,2)

        #data_info.object_pixels_avg.append(pixel_avg)
        data_info.object_area_avg[dirName] = area_avg

    #print 'data_info.object_pixels_avg:'
    #print data_info.object_pixels_avg
    print 'data_info.object_area_avg:'
    print data_info.object_area_avg
    return ret


def CalcAccuracySegDir(PredictInfo=None,setsPath=None,top=3,verbose=1,data_set_path=None):
    '''
    :param setsPath: images path
    :param top: The maximum probability of top categories
    :param verbose: verbosity mode, 0 or 1.
    :return:recognization accuracy
    '''
    totalNum=0
    totalErrList=[]
    setsPath += '/'
    for _, dirs, _ in os.walk(setsPath):
        break
    for dirName in dirs:
        #for _, _, files in os.walk(setsPath+dirName):
        n, errList = segImgDirforAcc(PredictInfo=PredictInfo,segPath=setsPath+dirName,data_set_path=data_set_path)
        totalNum += n
        totalErrList += errList
        #totalErrList.extend(errList)

    totalAcc = (totalNum-len(totalErrList))*1.0 / totalNum

    print 'seg err files:'
    for errfile in totalErrList:
        print errfile

    print
    print ("total accury is " + str(float('%.4f' % totalAcc )))
    return totalAcc


def compList(a,b):
    if len(a) == len(b):
        c = list(set(a).intersection(set(b))) #交集
        #list(set(a).union(set(b)))       #并集
        #list(set(b).difference(set(a)))  # b中有而a中没有的
        if len(c)==len(a):
            return True
    return False
#a = ['和鱼的', '是的']
#b = ['是的', '和鱼的']
#print compList(a, b)
#exit(0)

def segImgDirforAcc(segPath=None,PredictInfo=None,data_set_path=None):
    '''
    images in folder segmentation
    :param segPath: images path
    :return: print and plt show result
    '''
    judgeInfo = JudgeInfo()
    overArea = PredictInfo.IMG_ROW_OUT * PredictInfo.IMG_COL_OUT
    n=0
    errList = []
    for _, _, files in os.walk(segPath):
        break
    print
    print ("coming " + segPath)
    n = len(files)
    print 'files num is ' +str(n)
    categoryList=[]
    objects = open(segPath+"/category.txt")
    for line in objects:
        line =line[0:-1] #去掉最后的换行符
        categoryList.append(line)
    print 'category is:'
    print categoryList


    for file in files:

        if cmp(file, "category.txt") == 0:
            continue

        print ('predict  ' + file)
        img = data_label.loadImage(segPath + '/' + file,data_info=PredictInfo)
        y_p, pred = PredictInfo.model.predict(img)
        segImg,dishes_info = getRgbImgFromUpsampling(imgP=pred,data_info=PredictInfo)

        # display source img
        objectList=[]
        objectNum = 0
        for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
            xl = dish_v[2][0]
            yl = dish_v[2][1]
            xr = dish_v[2][2]
            yr = dish_v[2][3]
            h = xr - xl + 1
            w = yr - yl + 1
            area = h * w
            areaRate = dish_v[0] * 1.0 / overArea
            #fullness = dish_v[0] * 1.0 / area
            if areaRate > judgeInfo.tageCorrectTh: #and fullness > judgeInfo.degreeCorrectTh:
                objectNum += 1
                objectList.append(dish_v[1])
                #print("the %s mostly has found,AreaRate and fullness:(%f,%f)" % (dish_v[1], areaRate, fullness))

        if compList(objectList,categoryList)==False:
            errList.append(file)
            save_seg_img(file_name=file,img=img,seg_img=segImg,data_set_path=data_set_path)
            print "seg "+file+' err,NN found:'
            print objectList

    print (segPath + " accury is " + str(n-len(errList)) + '/' + str(n))
    return n, errList


def save_seg_img(file_name=None,img=None,seg_img=None,data_set_path=None):
    plt.figure(figsize=(6, 8))
    # display source img
    ax = plt.subplot(2, 1, 1)
    ax.imshow(img[0] / 255.)
    ax.set_title(os.path.basename(file_name))
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 1, 2)
    # display result
    ax.imshow(seg_img)
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt.savefig(out_path+'rgb_pixel.jpg')
    pixel_name = os.path.basename(file_name)
    pixel_name = pixel_name[0:-4]

    path = data_set_path + '/predictImg/err'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print 'mkdir ' + path
    plt.savefig(path +'/'+ pixel_name + '_pcla.jpg')
    # plt.show()
    plt.close()



# no use, deprecated.

def segImgfile(seg_model=None, data_info= None, url=None):
    '''
    image segmentation
    :param file: image path
    :return: print and plt show result
    '''
    judgeInfo = JudgeInfo()

    plt.figure(figsize=(20, 5))

    print ('predict  '+url)
    img = data_label.loadImage(url=url,data_info=data_info)
    y_p, pred = seg_model.predict(img)
    #print pred.shape()
    segImg,dishes_info = getRgbImgFromUpsampling(imgP=pred, data_info=data_info)

    # display source img
    ax = plt.subplot(2, 1, 1)
    ax.imshow(img[0]/255.)
    ax.set_title(url)
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 1, 2)
    foundSthFlag = 0
    for dish_k, dish_v in dishes_info.items():  # dish_v:(dot, name, (xl, yl, xr, yr))
        xl = dish_v[2][0]
        yl = dish_v[2][1]
        xr = dish_v[2][2]
        yr = dish_v[2][3]
        h = xr - xl + 1
        w=  yr - yl + 1
        area = h * w
        areaRate = dish_v[0]*1.0/(data_info.IMG_ROW_OUT * data_info.IMG_COL_OUT)
        fullness = dish_v[0]*1.0/area
        if areaRate > judgeInfo.tageCorrectTh and fullness >judgeInfo.degreeCorrectTh:
            #recognize something
            foundSthFlag = 1
            print("the %s mostly has found, number of pixels and fullness:(%d,%f)" % (dish_v[1], dish_v[0], fullness))
            central_x = (xr + xl)/2
            central_y = (yr + yl)/2
            #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
            #ax.text(xl, central_y, (dish_v[1].decode('utf8'))[-4:0], size=8, color="r", fontproperties=zhfont1)
    if foundSthFlag:
        # display result
        ax.imshow(segImg)
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def locateImgfile(url):
    '''
    Mark the location of the object with the rectangle of CV2
    :param url: images path
    :return: print and CV2 show image
    '''

    print ('predict  ' + url)
    img = loadImage(url)
    pred = segModel.predict(img)
    segImg, dishes_info = getRgbImgFromUpsampling(pred)
    bgrImg = segImg[:, :, ::-1]  # rgb to bgr
    cv2.imshow('category pixels', bgrImg)

    img = img[0]
    for dish_k, dish_v in dishes_info.items():  # dish_v:(dot, name, (xl, yl, xr, yr))
        xl = dish_v[2][0]
        yl = dish_v[2][1]
        xr = dish_v[2][2]
        yr = dish_v[2][3]
        h = xr - xl + 1
        w = yr - yl + 1
        area = h * w
        areaRate = dish_v[0] * 1.0 / overArea
        fullness = dish_v[0] * 1.0 / area
        if areaRate > tageCorrectTh and fullness > degreeCorrectTh:
            # recognize something
            print("the %s mostly has found, number of pixels and fullness:(%d,%f)" % (dish_v[1], dish_v[0], fullness))
            central_x = (xr + xl) / 2
            central_y = (yr + yl) / 2
            cv2.rectangle(img, (xl, yl), (xr, yr), (0, 255, 0), 2)
            cv2.putText(img, 'dish-' + str(dish_k), (central_x, central_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))

    img = img[:, :, ::-1]  # rgb to bgr
    cv2.imshow('location', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def predictImage(model=None,url=None):
    img = data_label.loadImage(url)

    if model == None :
        print 'please creatModel...'
        return

    preds = model.predict(img)
    return preds


def getTopObject(array, top=3):
    '''
    get top object in predict
    :param array: model predict result
    :param top: The maximum probability of top categories
    :return: object:['name', 'degree','index']
    '''
    object = []
    arrayTemp = array[0][:]
    for i in range(top):
        index = arrayTemp.argmax()
        for className, classIndex in classNameDic.items():
            if index == classIndex and arrayTemp[index] > 0.0001:
                object.append((className, float('%.4f' % arrayTemp[index]), index))

        arrayTemp[index] = 0
    return object


def predictImgSets(setsPath,top=3):
    '''
    images prediction
    :param setsPath: images path
    :param top: The maximum probability of top categories
    :return: img name, softmax rate,categories index
    '''
    setsPath += '/'
    for _, dirs, _ in os.walk(setsPath):
        for dirName in dirs:
            print
            print ("coming "+ dirName)
            for _, _, files in os.walk(setsPath+dirName):
                for name in files:
                    if cmp(file, "category.txt") == 0:
                        continue
                    print ('predict  '+name)
                    pred = predictImage(setsPath+dirName+'/'+name)
                    topObject = getTopObject(pred, top)
                    print (topObject)
                    print


def CalcAccuracyImgDir(setsPath,top=3,verbose=1):
    '''
    :param setsPath: images path
    :param top: The maximum probability of top categories
    :param verbose: verbosity mode, 0 or 1.
    :return:recognization accuracy
    '''
    totalNum=0
    totalAcc=0
    setsPath += '/'
    for _, dirs, _ in os.walk(setsPath):
        for dirName in dirs:
            print
            print ("coming "+ dirName)
            num = 0
            acc = 0
            for _, _, files in os.walk(setsPath+dirName):
                for name in files:
                    if cmp(file, "category.txt") == 0:
                        continue
                    #print 'predict'+name
                    num += 1
                    totalNum +=1
                    pred = predictImage(setsPath+dirName+'/'+name)
                    topObject = getTopObject(pred, top+4)

                    for i in range(top):
                        ret=cmp(dirName, topObject[i][0]) #0:name
                        if ret==0:
                            acc += 1
                            totalAcc +=1
                            break
                        elif i==top-1 and verbose==1:
                            print (name+" distinguish err")
                            print (topObject)
                            break

            print (dirName+" accury is "+str(acc)+'/'+str(num))   #+str(float('%.4f' % (acc*1.0/num)))

    print()
    print ("total accury is " + str(float('%.4f' % (totalAcc*1.0/totalNum))))


print 'out predic'