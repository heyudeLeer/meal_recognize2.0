#!/usr/bin/python
# -*- coding: utf-8 -*-

import meal_recognize2

from meal_recognize2 import pre_data_set
from meal_recognize2 import train_data_set
from meal_recognize2 import load_trained_model
from meal_recognize2 import predict_img


if __name__ == '__main__':

    restaurant_path = '/home/heyude/data_set/gpu_model/model'
    project_path = '/home/heyude/PycharmProjects/meal_recognize2'
    img_url = '/home/heyude/data_set/gpu_model/model/predictInfo/pixel_level3/webwxgetmsgimg'

    #pre_data_set(data_set_path=restaurant_path,project_path=project_path)

    ret,predict_model = train_data_set(data_set_path=restaurant_path,pixel_level=3)
    # ret : self_check status
    # 0 : success
    # 1:  warning
    #-1:  error

    if ret == 0 :
        predict_model = load_trained_model(data_set_path=restaurant_path,pixel_level=3)

        dict_str = predict_img(predict_info=predict_model,image_url=img_url)
        print dict_str

        # check
        dict_str = predict_img(image_url=img_url, predict_info=predict_model,check=True,data_set_path=restaurant_path)
        print dict_str



