# -*- coding: utf-8 -*-

import os
import sys
import gc

import keras
import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling2D,Conv2D

import time
import matplotlib.pyplot as plt
import cv2
import datetime

import train
import predic
import data_label

class DataInfo:
    def __init__(self):

        self.gpu_num = 1
        self.batch_size_base = 1
        self.confidence_threshold = 0.75
        self.threshold_value = np.uint8(self.confidence_threshold * 255)


        self.IMG_ROW = 512 #1024 #960  # 1920, 1080
        self.IMG_COL = 384 #768 #540

        self.CNN_OUT_DIV = None
        self.IMG_ROW_OUT = None
        self.IMG_COL_OUT = None
        self.enlarge_size = 0

        self.label_mode = 0
        self.load_pre_weights = False
        self.step_check = False

        self.epoch = 30
        self.one_hot_check = False
        self.boost_self_check = False
        self.edge_line = 0.9999           #使用时边界清晰 , median:0.999999/1e-08 , 0.9999/1e-06 , 0.99/1e-05
        self.edge_upper = 0.9
        self.edge_lower = 0.1

        self.base_model = None
        self.base_model_weight_file = None
        self.steps_per_epoch = 0

        self.one_hot_check_save_path = None
        self.boost_self_check_save_path = None

        self.start_get_val =False
        self.one_hot_x_val = None
        self.one_hot_y_val = None
        self.boost_x_val = None
        self.boost_label_val = None
        self.val_full = False
        self.median_th = 1e-4
        self.train_abs_loss_sum = 0.0
        self.loss_tabel=None
        self.sample_loss=0.0
        self.contour=None
        self.img_contour=None
        self.boost_finish=False

        self.big_loss_x = None
        self.big_loss_y = None
        self.big_loss_len = 0

                           #brightness_range, color_range, contrast_range, sharpness_range
        self.enhance_par = [  (0.7, 1.3),      (0.5, 1.5),  (0.5, 1.5),    (0.5, 1.5)]
        self.enhance_enable = False

        self.sess = None
        self.graph = None
        self.cpus = 1
        self.enhance_index = 0
        self.val_data_extend = 15       # 5x20 = 100
        self.train_data_extend = 5     # 5x10 = 50
        self.init=True
        self.pixel_level= 0
        self.batch_size_GPU = 1

        self.overlayerBg=True
        self.back_ground_path='/home/heyude/PycharmProjects/data_set/snacks2.0/black/bg'

        self.black_label_threash = 68

        self.class_num = 0
        self.class_name_dic_t = None
        self.train_img_num = 0
        self.train_generator = None
        self.val_img_num = 0
        self.val_generator = None
        self.val_datas = None
        self.model = None
        self.val_2Dlabel=None
        self.avg_class_num = 0.0
        self.avg_cce_loss = 0.0
        self.one_hot_var = 0.0
        self.contour_pixels = 0
        self.err_object=0.0

        #self.object_pixels_avg = []
        self.object_area_avg = {}
        self.data_gen=None

    def para_init(self, pixel_level=0):

        self.pixel_level = pixel_level

        if pixel_level == 2:
            self.IMG_ROW_OUT = self.IMG_ROW / 4
            self.IMG_COL_OUT = self.IMG_COL / 4

        elif pixel_level==3:
            self.IMG_ROW_OUT = self.IMG_ROW / 8
            self.IMG_COL_OUT = self.IMG_COL / 8

        elif pixel_level==4:
            self.IMG_ROW_OUT = self.IMG_ROW / 16
            self.IMG_COL_OUT = self.IMG_COL / 16

        else:
            print "error, pixel_level should in [2,3,4]"
            exit(0)


class PredictInfo:
    def __init__(self):
        self.name = None
        self.IMG_ROW = 0
        self.IMG_COL = 0
        self.IMG_ROW_OUT = 0
        self.IMG_COL_OUT = 0
        self.class_num = 0
        self.class_name_dic_t = None
        self.model = None
        self.header_model=None
        self.train_img_num = 0
        #self.object_pixels_avg = []
        self.object_area_avg = {}

        self.confidence_threshold = 0.75
        self.threshold_value = np.uint8(self.confidence_threshold * 255)


def saveStruct(*struct_datas):
    for datas in struct_datas:
        np.save(datas.name + "IMG_ROW", datas.IMG_ROW)
        np.save(datas.name + "IMG_COL", datas.IMG_COL)
        np.save(datas.name + "IMG_ROW_OUT", datas.IMG_ROW_OUT)
        np.save(datas.name + "IMG_COL_OUT", datas.IMG_COL_OUT)
        np.save(datas.name + "class_num", datas.class_num)
        np.save(datas.name + "class_name_dic_t", dict(datas.class_name_dic_t))
        np.save(datas.name + "train_img_num", datas.train_img_num)
        if len(datas.object_area_avg)>0:
            #np.save(datas.name + "object_pixels_avg", datas.object_pixels_avg)
            np.save(datas.name + "object_area_avg", datas.object_area_avg)
        print 'object_area_avg'
        print datas.object_area_avg
        #print 'object_pixels_avg'
        #print datas.object_pixels_avg

        print(datas.name + " info have saved!--------------")


def loadStruct(*struct_datas):
    for datas in struct_datas:

        datas.IMG_ROW = np.load(datas.name + "IMG_ROW.npy")
        datas.IMG_COL = np.load(datas.name + "IMG_COL.npy")
        datas.IMG_ROW_OUT  = np.load(datas.name + "IMG_ROW_OUT.npy")
        datas.IMG_COL_OUT = np.load(datas.name + "IMG_COL_OUT.npy")
        datas.class_num = np.load(datas.name + "class_num.npy")
        dict_array = np.load(datas.name + "class_name_dic_t.npy")
        datas.class_name_dic_t = dict_array.item()                ##### key point
        datas.train_img_num = np.load(datas.name + "train_img_num.npy")
        #datas.object_pixels_avg = np.load(datas.name + "object_pixels_avg.npy")
        datas.object_area_avg = np.load(datas.name + "object_area_avg.npy")
        datas.object_area_avg = datas.object_area_avg.item()

        print 'load object_area_avg'
        print type(datas.object_area_avg)
        print datas.object_area_avg
        #print 'object_pixels_avg'
        #print datas.object_pixels_avg

        print(datas.name + "* have loaded...!")


def get_thredhold_lower(data_info=None):
    #auto, multi: 1.2-->2.2
    th = 1.0 / data_info.class_num
    y = 2.2 - 2 * th  # ( [0.5,1.2] [0.1,2.0],[0.0000001,2.2])
    print 'threshold avg times is '+str(y)
    th = round(th * y * 255)


def train_data_set(data_set_path="/path/to/data_set/restaurant_name",pixel_level=3):
    '''
    :param data_set_path:
    :return:predict need info, and will be save in the data_set_path
    '''

    # pre data set
    #data_label.pre_data_set(data_set_path=data_set_path)

    # init global data
    dataInfo = DataInfo()
    dataInfo.para_init(pixel_level=pixel_level)
    # get class_num/class_name/img_num/data_gen and so on by path_data_set
    data_label.get_data_info(data_set_path=data_set_path, data_info=dataInfo)

    # save predict info in the data_set_path
    path = data_set_path + '/predictInfo/pixel_level'+ str(pixel_level)+ '/'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print 'mkdir ' + path

    graph = tf.get_default_graph()
    dataInfo.graph = graph
    dataInfo.sess = tf.Session(graph=graph)

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 现在

    if dataInfo.one_hot_check is True:
        # save self_check err imgs in the data_set_path
        self_check_path = data_set_path + '/predictInfo/pixel_level' + str(pixel_level) + '/self_check/one_hot/'+nowTime
        isExists = os.path.exists(self_check_path)
        if not isExists:
            os.makedirs(self_check_path)
            print 'mkdir ' + self_check_path
        dataInfo.one_hot_check_save_path = self_check_path
    if dataInfo.boost_self_check is True:
        # save self_check err imgs in the data_set_path
        self_check_path = data_set_path + '/predictInfo/pixel_level' + str(pixel_level) + '/self_check/one_hot_boost/'+nowTime
        isExists = os.path.exists(self_check_path)
        if not isExists:
            os.makedirs(self_check_path)
            print 'mkdir ' + self_check_path
        dataInfo.boost_self_check_save_path = self_check_path

    # train CNNs
    dataInfo.model = train.train_model(data_set_path=data_set_path,
                                                    data_info=dataInfo)  # save model weights on disk

    # get predict info
    predicInfo = PredictInfo()
    predicInfo.IMG_ROW = dataInfo.IMG_ROW
    predicInfo.IMG_COL = dataInfo.IMG_COL
    predicInfo.IMG_ROW_OUT = dataInfo.IMG_ROW_OUT
    predicInfo.IMG_COL_OUT = dataInfo.IMG_COL_OUT
    predicInfo.class_num = dataInfo.class_num
    predicInfo.class_name_dic_t = dataInfo.class_name_dic_t
    predicInfo.model = dataInfo.model
    predicInfo.train_img_num = dataInfo.train_img_num

    ret = predic.getPerPixels(setsPath=data_set_path + '/train', data_info=dataInfo,out_path=data_set_path)
    #predicInfo.object_pixels_avg = dataInfo.object_pixels_avg
    predicInfo.object_area_avg = dataInfo.object_area_avg

    predicInfo.name = path
    saveStruct(predicInfo, )
    lib_name = os.path.basename(data_set_path)
    #predicInfo.model.save(path + lib_name + '.h5') # save model on disk
    print '############# save ' + lib_name + ' predict need info in + data_set_path ######################'
    print

    return ret , predicInfo  # or error see log


# Once the model is created, you can test it with the following api
def creat_model(data_set_path="/path/to/data_set/restaurant_name", pixel_level=3):

    # load img info on disk by name
    predicInfo = PredictInfo()
    predicInfo.name = data_set_path + '/predictInfo/pixel_level'+ str(pixel_level)+'/'
    loadStruct(predicInfo)
    predicInfo.pixel_level = pixel_level
    header_model = train.creatXception(predicInfo, upsample=True,Div=16, train=False)
    #header_model.summary()
    predicInfo.header_model = header_model

    return predicInfo  # recognition_info


def load_trained_weights(predicInfo=None,data_set_path="/path/to/data_set/restaurant_name", pixel_level=3):

    x = predicInfo.header_model.output
    x = keras.layers.Conv2D(predicInfo.class_num, (1, 1), use_bias=False, name='conv_out')(x)

    seg_output = Activation('softmax', name='seg_out')(x)
    x = GlobalAveragePooling2D()(x)
    main_output = Activation('softmax', name='main_out')(x)
    predicInfo.model = Model(inputs=predicInfo.header_model.input, outputs=[main_output, seg_output], name='acting_model')

    weight_file = data_set_path + '/predictInfo/pixel_level' + str(pixel_level) + '/robust.hdf5'
    predicInfo.model.load_weights(weight_file)

    return predicInfo  # recognition_info


def predict_img(image_url='path/to/image',predict_info=None, check=False,data_set_path="/path/to/data_set/restaurant_name"):

    dic = predic.segImgfile_web(data_info=predict_info, url=image_url, out_path=data_set_path,show=check)
    # ret = predic.segImgDir(seg_model=model, data_info= predic_info, segPath='/home/heyude/temp/seg')

    return dic


def model_check(data_set_path=None, img_path=None):
    '''
    images in folder segmentation
    :param segPath: images path
    :return: print and plt show result
    '''
    #PredictInfo_0 = load_trained_model(data_set_path,pixel_level=0)
    #PredictInfo = load_trained_model(data_set_path)

    PredictInfo_4 = creat_model(data_set_path=data_set_path, pixel_level=2)
    PredictInfo_4 = load_trained_weights(predicInfo=PredictInfo_4, data_set_path=data_set_path, pixel_level=2)

    PredictInfo = creat_model(data_set_path=data_set_path, pixel_level=3)
    PredictInfo = load_trained_weights(predicInfo=PredictInfo, data_set_path=data_set_path, pixel_level=3)

    for _, dirs, _ in os.walk(img_path):
        break

    for dir_name in dirs:
        print ("coming " + img_path+'/'+dir_name)
        for _, _, files in os.walk(img_path+'/'+dir_name):
            break
        n = len(files)
        i = 0
        plt.figure(figsize=(20, 8))
        print 'files num is ' + str(n)

        for file in files:
            print
            url = img_path+'/'+dir_name + '/' + file
            print ('predict  '+url)
            img_4 = data_label.loadImage(url=url,data_info=PredictInfo_4)
            img = data_label.loadImage(url=url,data_info=PredictInfo)

            _, pred_4 = PredictInfo_4.model.predict(img_4)
            _, pred = PredictInfo.model.predict(img)

            RgbImg_4 = predic.get_rgb_mark(imgP=pred_4, data_info=PredictInfo_4)
            dishes_info_4, seg_contour_4 = predic.get_dishes_with_confidence_debug(dataInfo=PredictInfo_4, segImg=pred_4[0])

            RgbImg = predic.get_rgb_mark(imgP=pred, data_info=PredictInfo)
            dishes_info, seg_contour = predic.get_dishes_with_confidence_debug(dataInfo=PredictInfo, segImg=pred[0])

            i += 1
            # display source img
            ax = plt.subplot(5, n, i)
            ax.set_title(file[-8:-1])
            ax.imshow(img[0]/255.)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(5, n, i+n)
            ax.set_title('base')
            # display result
            ax.imshow(RgbImg_4)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(5, n, i + n*2)
            # display result
            ax.imshow(RgbImg)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            #img = data_label.loadImage(url=plca_path + '/' + shotname+'_pcla.jpg',data_info=PredictInfo)
            ax = plt.subplot(5, n, i+n*3)
            ax.set_title('base')
            ax.imshow(np.mean(seg_contour_4,axis=2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(5, n, i + n * 4)
            ax.imshow(np.mean(seg_contour,axis=2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
        plt.close()







