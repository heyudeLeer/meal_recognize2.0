#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import datetime
from img_segment import pixclass
from img_segment import predic
from img_segment import data_label


def train(restaurant_path=None):

    model_info = pixclass.train_data_set(data_set_path=restaurant_path,pixel_level=3)
    return model_info


def predict(restaurant_path=None,img_url=None):

    dict, model_info = pixclass.cold_predict_img(data_set_path=restaurant_path,image_url=img_url,pixel_level=3)
    print(dict)
    return model_info


def hot_test(restaurant_path=None,img_url=None,model_info=None):

    ret = pixclass.hot_predict_img(data_set_path=restaurant_path,
                                  image_url=img_url,
                                  predict_info=model_info)
    print(ret)
    return
    # private acc test
    print (model_info.class_name_dic_t)
    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test1')

    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test00')
    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test01')
    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test02')

    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test0')
    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test1')
    predic.segImgDir(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/test/test2')

    # predic.CalcAccuracySegDir(PredictInfo=model_info, setsPath=restaurant_path + '/predictImg/test/test_acc',
    #                          data_set_path=restaurant_path)

    predic.imgPredict2DShow(url=img_url, predict_info=model_info)


def test_hot_predict_by_dir(segPath=None,restaurant_path=None,model_info=None):
    print segPath
    for _, _, files in os.walk(segPath):
        print ("coming " + segPath)
        break
    n = len(files)
    print 'files num is ' + str(n)

    for file in files:

        if cmp(file, "category.txt") == 0:
            continue
        img_url = segPath + '/' + file
        ret = pixclass.hot_predict_img(data_set_path=restaurant_path,
                                       image_url=img_url,
                                       predict_info=model_info)
        print(ret)
        print


if __name__ == '__main__':

    restaurant_path = 'data_set/gpu_model/model'
    img_url = 'data_set/test0817/predictImg/IMG_0515.jpg'
    home = os.path.expandvars('$HOME')
    restaurant_path = home + '/' + restaurant_path
    img_url = home + '/' + img_url

    img_url1 = '/home/heyude/data_set/model/predictImg/img/fdbd7eda-14e3-4aa4-88f5-1dd4252a6c2f.jpg'
    img_url2 = '/home/heyude/data_set/model/predictImg/img/78288410-6ecd-4fed-ac51-0e472a078ce0.jpg'

    #img_url1 = '/home/heyude/data_set/gpu_model/model/predictImg/img/single/IMG_20181121_150918.jpg'
    #img_url2 = '/home/heyude/data_set/gpu_model/model/predictImg/img/single/IMG_20181121_150855.jpg'
    #img_url2 = '/home/heyude/data_set/gpu_model/model/predictImg/img/single/IMG_20181121_150855.jpg'

    #img_url1 = '/home/heyude/data_set/gpu_model/model/predictImg/img/sample/IMG_20181121_143818.jpg'
    #img_url2 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150055.jpg'

    #img_url1 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150055.jpg'
    #img_url2 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150145.jpg'
    img_url3 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150000.jpg'
    img_url4 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150231.jpg'
    img_url5 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150159.jpg'
    img_url6 = '/home/heyude/data_set/gpu_model/model/predictImg/img/pingpan_s/IMG_20181121_150009.jpg'





    #data_label.filter_xy(path='/home/heyude/PycharmProjects/data_set/test0817/predictImg/test')
    #data_label.pre_data_set(data_set_path=restaurant_path) #only one time for every data_set
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print nowTime
    model_info=train(restaurant_path=restaurant_path)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print nowTime
    model_info=predict(restaurant_path=restaurant_path,img_url=img_url3)
    #hot_test(restaurant_path=restaurant_path, img_url=img_url4, model_info=model_info)
    test_hot_predict_by_dir(restaurant_path=restaurant_path, segPath=restaurant_path + '/predictImg/img/pingpan_hot_test', model_info=model_info)


    #predic.imgPredict2DShow_diff(url=[img_url1,img_url2], predict_info=model_info)
    #predic.imgPredict2DShow_diff(url=[img_url3,img_url4], predict_info=model_info)
    #predic.imgPredict2DShow_diff(url=[img_url6,img_url5], predict_info=model_info)

    #predic.imgPredict2DShow(url=img_url, predict_info=model_info)


    #hot_test(restaurant_path=restaurant_path, img_url=img_url, model_info=model_info)
    #predic.segImgDir_cantai(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/img/sample',plca_path=restaurant_path + '/predictImg/pcla')
    #predic.segImgDir_cantai(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/img/Baicaidunrou',plca_path=restaurant_path + '/predictImg/pcla')
    predic.segImgDir_cantai(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/img/pingpan_s',plca_path=restaurant_path + '/predictImg/pcla')
    #predic.segImgDir_cantai(PredictInfo=model_info, segPath=restaurant_path + '/predictImg/img/pingpan',plca_path=restaurant_path + '/predictImg/pcla')






