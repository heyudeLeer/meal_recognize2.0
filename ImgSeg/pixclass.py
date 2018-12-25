# -*- coding: utf-8 -*-

import os
import sys
import gc

import numpy as np
from keras.models import load_model
from time import sleep, time
import tensorflow as tf

import train
import predic
import data_label

class DataInfo:
    def __init__(self):
        #self.IMG_ROW = 384
        #self.IMG_COL = 384
        self.IMG_ROW = 512 #1024 #960  # 1920, 1080
        self.IMG_COL = 384 #768 #540

        self.CNN_OUT_DIV = None
        self.IMG_ROW_OUT = None
        self.IMG_COL_OUT = None
        self.REDUCE_TIME = 0
        self.batch_size_GPU = 0
        self.enlarge_size = 0

        self.label_mode = 0
        self.load_pre_weights = False
        self.step_check = False

        self.epoch = 30
        self.CCE = True
        self.ont_hot_check = False
        self.boost_self_check = False

        self.ont_hot_check_save_path = None
        self.boost_self_check_save_path = None

        self.start_get_val =False
        self.one_hot_x_val = None
        self.one_hot_y_val = None
        self.boost_x_val = None
        self.boost_label_val = None

                           #brightness_range, color_range, contrast_range, sharpness_range
        self.enhance_par = [  (0.75, 1.15),      (0.8, 1.4),  (0.8, 1.4),    (0.8, 1.4)]
        self.enhance_enable = True

        self.sess = None
        self.cpus = 1
        self.enhance_index = 0
        self.val_data_extend = 20       # 5x20 = 100
        self.train_data_extend = 10     # 5x10 = 50
        self.init=True
        self.pixel_level= 0

        self.strenghthen_train_flag = False
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
        self.object_pixels_avg = []
        self.object_area_avg = []

    def para_init(self, pixel_level=0):

        self.pixel_level = pixel_level

        if pixel_level==0:
            self.CNN_OUT_DIV = None
            self.IMG_ROW /= 2
            self.IMG_COL /= 2
            self.IMG_ROW_OUT = self.IMG_ROW / 32
            self.IMG_COL_OUT = self.IMG_COL / 32
            self.REDUCE_TIME = 5
            self.enlarge_size = 4
            self.batch_size_GPU = 8

        elif pixel_level==1:
            self.CNN_OUT_DIV = 32
            self.IMG_ROW_OUT = self.IMG_ROW / 32
            self.IMG_COL_OUT = self.IMG_COL / 32
            self.REDUCE_TIME = 5
            self.batch_size_GPU = 5

        elif pixel_level==2:
            self.CNN_OUT_DIV = 16
            self.IMG_ROW_OUT = self.IMG_ROW / 16
            self.IMG_COL_OUT = self.IMG_COL / 16
            self.REDUCE_TIME = 4
            self.batch_size_GPU = 3

        elif pixel_level==3:
            self.CNN_OUT_DIV = 8
            self.IMG_ROW_OUT = self.IMG_ROW / 8
            self.IMG_COL_OUT = self.IMG_COL / 8
            self.REDUCE_TIME = 3
            self.batch_size_GPU = 3

        else:
            print "error, pixel_level should in [0,3]"
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
        self.train_img_num = 0
        self.object_pixels_avg = []
        self.object_area_avg = []


def saveStruct(*struct_datas):
    for datas in struct_datas:
        np.save(datas.name + "IMG_ROW", datas.IMG_ROW)
        np.save(datas.name + "IMG_COL", datas.IMG_COL)
        np.save(datas.name + "IMG_ROW_OUT", datas.IMG_ROW_OUT)
        np.save(datas.name + "IMG_COL_OUT", datas.IMG_COL_OUT)
        np.save(datas.name + "class_num", datas.class_num)
        np.save(datas.name + "class_name_dic_t", dict(datas.class_name_dic_t))
        np.save(datas.name + "train_img_num", datas.train_img_num)
        np.save(datas.name + "object_pixels_avg", datas.object_pixels_avg)
        np.save(datas.name + "object_area_avg", datas.object_area_avg)
        print 'object_area_avg'
        print datas.object_area_avg
        print 'object_pixels_avg'
        print datas.object_pixels_avg

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
        datas.object_pixels_avg = np.load(datas.name + "object_pixels_avg.npy")
        datas.object_area_avg = np.load(datas.name + "object_area_avg.npy")

        print 'object_area_avg'
        print datas.object_area_avg
        print 'object_pixels_avg'
        print datas.object_pixels_avg

        print(datas.name + "* have loaded...!")


def train_data_set(data_set_path="/path/to/data_set/restaurant_name",pixel_level=0):
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

    with tf.Session() as sess:
        dataInfo.sess = sess

    if dataInfo.ont_hot_check is True:
        # save self_check err imgs in the data_set_path
        self_check_path = data_set_path + '/predictInfo/pixel_level' + str(pixel_level) + '/self_check/one_hot'
        isExists = os.path.exists(self_check_path)
        if not isExists:
            os.makedirs(self_check_path)
            print 'mkdir ' + self_check_path
        dataInfo.ont_hot_check_save_path = self_check_path
    if dataInfo.boost_self_check is True:
        # save self_check err imgs in the data_set_path
        self_check_path = data_set_path + '/predictInfo/pixel_level' + str(pixel_level) + '/self_check/one_hot_boost'
        isExists = os.path.exists(self_check_path)
        if not isExists:
            os.makedirs(self_check_path)
            print 'mkdir ' + self_check_path
        dataInfo.boost_self_check_save_path = self_check_path

    # train CNNs
    dataInfo.model = train.train_model(data_set_path=data_set_path,
                                                    data_info=dataInfo)  # save model weights on disk
    if dataInfo.strenghthen_train_flag is True:
        # strengthen train by one-hot segmodel
        dataInfo.model = train.train_model_by_predict(data_set_path=data_set_path,
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

    predic.getPerPixels(setsPath=data_set_path + '/train', data_info=dataInfo)
    predicInfo.object_pixels_avg = dataInfo.object_pixels_avg
    predicInfo.object_area_avg = dataInfo.object_area_avg

    predicInfo.name = path
    saveStruct(predicInfo, )
    lib_name = os.path.basename(data_set_path)
    predicInfo.model.save(path + lib_name + '.h5') # save model on disk
    print '############# save ' + lib_name + ' predict need info in + data_set_path ######################'
    print

    #del trainModel, segModel  # deletes the existing model
    #gc.collect()
    #sleep(20)
    #print 'img_rec finsh ,should exa source!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    return predicInfo  # or error see log


# Once the model is created, you can test it with the following api
def cold_predict_img(data_set_path="/path/to/data_set/restaurant_name", image_url='path/to/image',pixel_level=0,check=False):
    '''
    :param data_set_path:
    :param image_url:
    :return:a dic and pcla.jpg in data_set_path/predictImg of predict result; predict_info for hot_predict,
    '''

    lib_name = os.path.basename(data_set_path)
    # load model on disk by name
    model = load_model(filepath= data_set_path +'/predictInfo/pixel_level'+ str(pixel_level) +'/'+lib_name +'.h5')
    #model.summary()


    # load img info on disk by name
    predicInfo = PredictInfo()
    predicInfo.name = data_set_path + '/predictInfo/pixel_level'+ str(pixel_level)+'/'
    loadStruct(predicInfo)
    predicInfo.model = model

    #print "IMG_ROW:" + str(predicInfo.IMG_ROW)
    #print "IMG_COL:" + str(predicInfo.IMG_COL)
    #print "IMG_ROW_OUT:" + str(predicInfo.IMG_ROW_OUT)
    #print "IMG_COL_OUT:" + str(predicInfo.IMG_COL_OUT)
    #print "class_num:" + str(predicInfo.class_num)
    #print predicInfo.class_name_dic_t
    #print 'predict model is '
    #print type(model)


    dic = predic.segImgfile_web(data_info= predicInfo, url=image_url,out_path=data_set_path,show=check) # save pcla.jpg in data_set_path/predictImg
    #ret = predic.segImgDir(seg_model=model, data_info= predicInfo, segPath='/home/heyude/temp/seg')

    return dic, predicInfo  # recognition_info


def hot_predict_img(data_set_path="/path/to/data_set/restaurant_name", image_url='path/to/image',predict_info=None,check=False):

    # load model on disk by name
    #model = predict_info.model
    #model.summary()
    #print 'predict by '
    #print type(model)

    #lib_name = os.path.basename(data_set_path)
    #print 'restaurant_name is ' + lib_name
    #print
    # load img info on disk by name
    #print 'the hot model info is:'
    #print 'restaurant_name is ' + predict_info.name
    #print "IMG_ROW:" + str(predict_info.IMG_ROW)
    #print "IMG_COL:" + str(predict_info.IMG_COL)
    #print "IMG_ROW_OUT:" + str(predict_info.IMG_ROW_OUT)
    #print "IMG_COL_OUT:" + str(predict_info.IMG_COL_OUT)
    #print "class_num:" + str(predict_info.class_num)
    #print predict_info.class_name_dic_t

    dic = predic.segImgfile_web(data_info=predict_info, url=image_url, out_path=data_set_path,show=check)
    # ret = predic.segImgDir(seg_model=model, data_info= predic_info, segPath='/home/heyude/temp/seg')

    return dic  # recognition_info







