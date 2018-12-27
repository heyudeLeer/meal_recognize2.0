
# encoding: utf-8
print 'in data_label'


import os
from PIL import ImageFile
import datetime
import math


try:
    from PIL import ImageEnhance
    from PIL import ImageStat
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
import uniout
import cv2
from keras import backend as K
from keras.preprocessing import image
import random


def get_session(gpu_fraction=0.8):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(get_session(0.8))


def pre_data_set(data_set_path=None):

    ### split data set to train and validation
    project_path = os.path.abspath('.')  # 获得当前工作目录
    split_tool = project_path + '/tools/split.sh'
    os.system("cp %s %s " % (split_tool,data_set_path))  # cp split.sh to data_set_path
    lib_name  = os.path.basename(data_set_path)
    temp_path = data_set_path + '/' + lib_name
    print 'data set img path ' + temp_path
    filter_xy(path=temp_path)
    os.system("cd %s; ./split.sh %s 0.0" % (data_set_path, temp_path))
    print 'split %s to train and validation' % temp_path

def filter_xy(path=None):   # shape to same

    for _, dirs, _ in os.walk(path):
        for dirName in dirs:
            print
            print ("coming "+ dirName)
            for _, _, files in os.walk(path+'/'+dirName):
                break
            n = len(files)
            print 'files num is ' + str(n)

            for file in files:
                url = path + '/' + dirName + '/' + file
                img = cv2.imread(url)
                h,w,z = img.shape
                if w > h:
                    img = np.transpose(img, (1, 0, 2))
                    print 'transpose '
                    print url
                    print img.shape

                    cv2.imwrite(url,img)

# this is the augmentation configuration we will use for training
def get_data_info(data_set_path=None, data_info=None):

    print 'train data init...'
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=20.0,
        zoom_range=[0.7,1.3],
        #channel_shift_range=0.5, # No effect
        #brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip=True,
        #fill_mode='nearest'
    )
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    data_info.train_generator = datagen.flow_from_directory(
        data_set_path + '/train' ,  # this is the target directory
        target_size=(data_info.IMG_ROW, data_info.IMG_COL),  # all images will be resized to 150x150
        batch_size=data_info.batch_size_GPU,
        #save_to_dir=data_set_path+'/train_gen', save_prefix='good', save_format='jpg',
        shuffle=True,
        class_mode='categorical'
    )
    # get train labs info and item name
    classNameDic = data_info.train_generator.class_indices
    data_info.class_name_dic_t = dict((v, k) for k, v in classNameDic.items())
    print data_info.class_name_dic_t
    data_info.train_img_num = data_info.train_generator.samples
    data_info.class_num = data_info.train_generator.num_classes
    #x, y = train_generator.next()

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    #datagen_val = ImageDataGenerator()
    data_info.val_generator = datagen.flow_from_directory(
        data_set_path + '/train',  # this is the target directory : validation
        target_size=(data_info.IMG_ROW, data_info.IMG_COL),  # all images will be resized to 150x150
        batch_size=data_info.batch_size_GPU*8,
        #save_to_dir=data_set_path+'/val_gen', save_prefix='shear', save_format='jpg',
        shuffle=True,
        class_mode='categorical'
    )
    data_info.val_img_num = data_info.val_generator.samples
    print 'train data init finsh'


def enhance_by_random(x=None, data_info=None,if_save=False ):

    pars = data_info.enhance_par
    brightness_range = pars[0]
    color_range = pars[1]
    contrast_range = pars[2]
    sharpness_range = pars[3]
    #print brightness_range,color_range,contrast_range,sharpness_range

    x = array_to_img(x)
    index = data_info.enhance_index
    name = []

    if index == 0 or index > 5:
        if len(brightness_range) != 2:
            raise ValueError('`brightness_range should be tuple or list of two floats. '
                             'Received arg: ', brightness_range)
        imgenhancer_Brightness = ImageEnhance.Brightness(x)
        u = np.random.uniform(brightness_range[0], brightness_range[1])
        #stat = ImageStat.Stat(x)
        #brightness = max(stat.mean)
        #print brightness
        x = imgenhancer_Brightness.enhance(u)
        name.append('_bright' + str(round(u,2)))

    if index == 1 or index > 5:
        if len(color_range) != 2:
            raise ValueError('`color_range should be tuple or list of two floats. '
                             'Received arg: ', color_range)
        imgenhancer_Color = ImageEnhance.Color(x)
        u = np.random.uniform(color_range[0], color_range[1])
        x = imgenhancer_Color.enhance(u)
        name.append('_color' + str(round(u,2)))

    if index == 2 or index > 5:
        if len(contrast_range) != 2:
            raise ValueError('`contrast_range should be tuple or list of two floats. '
                             'Received arg: ', contrast_range)
        imgenhancer_Contrast = ImageEnhance.Contrast(x)
        u = np.random.uniform(contrast_range[0], contrast_range[1])
        x = imgenhancer_Contrast.enhance(u)
        name.append('_contrast' + str(round(u,2)))

    if index == 3 or index > 5:
        if len(sharpness_range) != 2:
            raise ValueError('`sharpness_range should be tuple or list of two floats. '
                             'Received arg: ', sharpness_range)
        imgenhancer_Sharpness = ImageEnhance.Sharpness(x)
        u = np.random.uniform(sharpness_range[0], sharpness_range[1])
        x = imgenhancer_Sharpness.enhance(u)
        name.append('_sharp' + str(round(u,2)))

    if if_save is True and len(name)>0:
        name = ''.join(name)
        #print name
        home = os.path.expandvars('$HOME')
        save_path = home + '/data_set/enhance/'
        isExists = os.path.exists(save_path)
        if not isExists:
            os.makedirs(save_path)
            print 'mkdir ' + save_path
        rand_index = np.random.uniform(0, 10)
        if index+2 > rand_index:
            x.save(save_path + name + ".jpg") #随机观察

    index += 1
    if index == 7: #[0,3][4,5:No][>=6:All]
        index = 0

    data_info.enhance_index = index

    x = img_to_array(x)

    return x


def loadImage(url=None, use_cv2=True,data_info=None):

    if use_cv2 is False:
        img = image.load_img(url)#,target_size=(data_info.IMG_ROW,data_info.IMG_COL))
        img = image.img_to_array(img)
        h, w, z = img.shape
        if w > h:
            img = np.transpose(img, (1, 0, 2))
            print 'transpose '
            print url

        x = cv2.resize(img, (data_info.IMG_COL, data_info.IMG_ROW),interpolation=cv2.INTER_AREA)  # attention cv2 imgCols/imgRows exchange
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #x = np.array(x) / 255.
        return x
    else:
        y = cv2.imread(url)
        h, w, z = y.shape
        if w > h:
            y = np.transpose(y, (1, 0, 2))
            print 'transpose '
            print url

        y = cv2.resize(y, (data_info.IMG_COL, data_info.IMG_ROW), interpolation=cv2.INTER_AREA) # attention cv2 imgCols/imgRows exchange
        y = np.expand_dims(y, axis=0)
        y = y[:, :, :, ::-1]
        #cv2.imshow('imgcv2', y[0])
        #cv2.waitKey(0)
        # y = np.array(y) / 255.
        return y

    # deprecated, use PIL
    img = [np.transpose(scipy.misc.imresize(scipy.misc.imread(url), (imgRows, imgCols)), (0, 1, 2))]
    scipy.misc.imshow(img[0])
    return img


def mean_CCE_loss(y_true=0, y_pred=0,data_info=None):
    #logits_scaled = tf.nn.softmax(logits)
    y_pred = tf.convert_to_tensor(y_pred, dtype=np.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=np.float32)
    ret = K.categorical_crossentropy(y_true, y_pred)

    ret = (data_info.sess.run(ret))

    return ret


# by open_cvision tools
def train_generator_with2Dlabel(data_info=None, pure_2d=False): #by open_cvision tools

    for x, y in data_info.train_generator:

        #print 'x type'
        #print x.shape
        #print(type(x[0]), x[0].dtype, np.min(x[0]), np.max(x[0]))
        sample_len = len(y)
        labsBuf = np.zeros((sample_len, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, data_info.class_num), dtype=np.float32)
        if data_info.overlayerBg:
            mix_img = np.zeros((sample_len, data_info.IMG_ROW, data_info.IMG_COL, 3), dtype=np.uint8)

        for i in range(sample_len):
            ylist = list(y[i])
            index = ylist.index(1.0)

            if data_info.class_name_dic_t[index] == 'bg_no': # 抑制背景，效果还不错，也够简单
                y[i][index] = 0
                continue

            # img to label
            label,orign_size_label=img2laebl(img=x[i],data_info=data_info)

            labsBuf[i, :, :, index] = label

            if data_info.step_check:
                labels2DShow(img=x[i], labels=labsBuf[i], data_info=data_info)

            if data_info.overlayerBg:
                mix_img[i] = imgOverlayBg(img=x[i], label2D=orign_size_label, bg_path=data_info.back_ground_path, data_info=data_info)

        if data_info.overlayerBg:
            x=mix_img

        if data_info.label_mode == 1:
            yield x, labsBuf
        elif data_info.label_mode == 2:
            yield (x, {'main_out': y, 'seg_out': labsBuf})


# to train by GPU size,
# yield to a generator
def predict_2Dlabel_generator(data_info=None):
    length = data_info.train_img_num* data_info.val_data_extend
    data_info.boost_x_val = np.zeros((length, data_info.IMG_ROW, data_info.IMG_COL, 3),dtype=np.float32)
    data_info.boost_label_val = np.zeros((length, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT,data_info.class_num),dtype=np.float32)
    k = 0
    avg_p = 1.0 / data_info.class_num
    print 'boost sync...'
    if data_info.batch_size_GPU > 64:
        buf_len = data_info.batch_size_GPU
    else:
        buf_len = 64

    labsBuf = np.zeros((buf_len, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, data_info.class_num),dtype=np.float32)

    while True:

        if k == 0:   #do not know why?
            k += 1
            x = np.expand_dims(data_info.boost_x_val[0], axis=0)
            data_info.model.predict(x)
            yield None

        x, y  = data_info.train_generator.next()
        sample_len = len(y)
        for i in range(sample_len):

            if data_info.enhance_enable is True:
                x[i] = enhance_by_random(x=x[i], data_info=data_info)

            ylist = list(y[i])
            index = ylist.index(1.0)
            if data_info.class_name_dic_t[index] == 'bg' :  # 抑制背景，效果还不错，也够简单
                labsBuf[i] = avg_p
            else:
                img = np.expand_dims(x[i], axis=0)
                y_p, label_vect = data_info.model.predict(img)
                label = strengthen_method_median_power_tristate(label=label_vect[0], index=index, data_info=data_info)
                labsBuf[i] = label

            if data_info.start_get_val is True:
                if k == 1:
                    print(type(data_info.boost_label_val), data_info.boost_label_val.shape,
                          data_info.boost_label_val.dtype, np.min(data_info.boost_label_val), np.max(data_info.boost_label_val))
                if k < length:
                    data_info.boost_x_val[k-1] = x[i]
                    data_info.boost_label_val[k-1] = labsBuf[i]
                if k == length:
                    data_info.start_get_val = False
                    print(type(data_info.boost_x_val), data_info.boost_x_val.shape,
                          data_info.boost_x_val.dtype, np.min(data_info.boost_x_val), np.max(data_info.boost_x_val))
                    print(type(data_info.boost_label_val), data_info.boost_label_val.shape,
                          data_info.boost_label_val.dtype, np.min(data_info.boost_label_val),
                          np.max(data_info.boost_label_val))
                print k
                k += 1


        yield (x, labsBuf[0:sample_len])


def mean_squared_error(y_true=0, y_pred=0):
    return np.mean(np.square(y_pred - y_true))


def one_hot_self_check(model=None,data_info=None,thickness=1e-4,extend=10):

    k = 0
    bg_num = 0
    avg_p = 1.0 / data_info.class_num
    print '... self_check ...'
    print thickness
    data_info.model = model
    for x, y in data_info.train_generator:
        sample_len = len(y)
        for i in range(sample_len):

            k += 1
            ylist = list(y[i])
            index = ylist.index(1.0)
            if data_info.class_name_dic_t[index] == 'bg':
                print 'this is bg'
                bg_num += 1
                y[i, :] = avg_p
                thick = math.log(data_info.class_num) + thickness
            else:
                thick = thickness

            img = np.expand_dims(x[i], axis=0)
            y_p, seg = data_info.model.predict(img)

            main_out_loss = mean_CCE_loss(y_true=y[i], y_pred=y_p,data_info=data_info)
            loss = main_out_loss[0]

            if loss > thick:
                print 'err sample...'
                print loss
                print y[i]
                print y_p

                titels = []
                if data_info.CCE is True:
                    activation = 'softmax'
                else:
                    activation = 'sigmoid'
                title_up1 = 'one_hot: '+activation
                title_up2 = 'm_o/s_o/label'
                title_down = 'main_out_loss: '+ str(format(loss,'.3e'))
                titels.append(title_up1)
                titels.append(title_up2)
                titels.append(title_down)

                x_labels = []
                for index in range(data_info.class_num):
                    seg_mean = np.mean(seg[0, :, :, index])
                    x_labels.append(str(format(y_p[0,index],'.2e'))+'/'+str(round(seg_mean,2))+'/'+str(round(y[i,index],2)))

                labels2DShow(img=x[i],labels=seg[0],data_info=data_info,titels=titels, show=False,
                             x_labels=x_labels,save_path=data_info.ont_hot_check_save_path)

        print 'check '+str(k) + ' imgs...'
        print
        if k >= data_info.val_img_num * extend:
            print 'self check finish'
            print 'bg img are ' + str(bg_num)
            break


def predict_2Dlabel_datas_self_check(data_info=None, generator=None,thickness=1e-5):  # produce one time and save in mem, maybe for fit
    extend = data_info.train_data_extend
    x_val = np.zeros((data_info.train_img_num * extend, data_info.IMG_ROW, data_info.IMG_COL, 3), dtype=np.float32)
    y_val = np.zeros((data_info.train_img_num * extend, data_info.class_num), dtype=np.float32)
    labsBuf = np.zeros((data_info.train_img_num * extend, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, data_info.class_num),
                       dtype=np.float32)
    k = 0
    avg_p = 1.0 / data_info.class_num
    print 'generator 2D datas self_check...'
    print 'thickness is ' +str(thickness)
    for x, y in generator:
        sample_len = len(y)
        for i in range(sample_len):

            ylist = list(y[i])
            index = ylist.index(1.0)
            if data_info.class_name_dic_t[index] == 'bg':  # 抑制背景，效果还不错，也够简单
                y[i, :] = avg_p

            #if data_info.enhance_enable is True:
            #    x[i] = enhance_by_random(x=x[i], data_info=data_info)

            img = np.expand_dims(x[i], axis=0)
            y_p,label_vect = data_info.model.predict(img)

            if data_info.class_name_dic_t[index] == 'bg':  # 抑制背景，效果还不错，也够简单
                label = np.zeros((data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT,data_info.class_num),dtype=np.float32)
                label[:,:,:] = avg_p
            else:
                label_cp = label_vect[0].copy()
                label = strengthen_method_median_power_tristate(label=label_cp,index=index,data_info=data_info)

            labsBuf[k] = label
            x_val[k] = x[i]
            y_val[k] = y[i]
            k += 1

            if data_info.step_check:
                #print 'index of 2D_label after strengthen_method...'
                #print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index]
                #if index != 0:
                #    print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index-1]
                labels2DShow(img=x_val[k-1], labels=labsBuf[k-1], data_info=data_info)

                #print 'start val ....'
                #print(type(x), x.shape, x.dtype, np.min(x), np.max(x))
                #print(type(y_p), y_p.shape, y_p.dtype, np.min(y_p), np.max(y_p))
                #print(type(labsBuf), labsBuf.shape, labsBuf.dtype, np.min(labsBuf), np.max(labsBuf))

            loss = mean_CCE_loss(y_true=label, y_pred=label_vect[0],data_info=data_info)

            print
            print y[i]
            print y_p
            print(type(loss), loss.shape, loss.dtype, np.min(loss), np.max(loss))
            print loss[data_info.IMG_ROW_OUT / 2, :]
            loss_m = np.mean(loss)
            print 'loss mean: ' + str(loss_m)

            titels = []
            #title_up1 = 'boost_one_hot: ' + 'sigmoid_BCE'
            title_up1 = 'boost: ' + 'softmax_ccs'
            title_up2 = 'm_o/m_l'
            title_down = 'ccs_loss: ' + str(format(loss_m, '.3e'))
            titels.append(title_up1)
            titels.append(title_up2)
            titels.append(title_down)
            print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index]

            x_labels = []
            for i_ch in range(data_info.class_num):
                x_labels.append( str(round(y_p[0, i_ch], 3)) + '/' + str(round(y[i, i_ch], 3)))
                #+ str(round(loss[i_ch], 3)))


            labels2DShow_boost(img=x[i], labels=label, seg=label_vect[0], data_info=data_info, titels=titels, show=False,
                         x_labels=x_labels, save_path=data_info.boost_self_check_save_path)

            if k == data_info.train_img_num * extend:
                break

        print k
        if k ==  data_info.train_img_num * extend:
            print '2D datas are....'
            print(type(x_val), x_val.shape, x_val.dtype)
            print(type(y_val), y_val.shape, y_val.dtype)
            print(type(labsBuf), labsBuf.shape, labsBuf.dtype)

            if data_info.label_mode == 2:
                return (x_val, {'main_out': y_val, 'seg_out': labsBuf})
            else:
                return (x_val, labsBuf)


def predict_2Dlabel_datas_no_check(data_info=None, generator=None):# produce one time and save in mem, maybe for fit
    extend = data_info.train_data_extend
    x_val = np.zeros((data_info.train_img_num * extend, data_info.IMG_ROW, data_info.IMG_COL, 3),
                     dtype=np.float32)
    y_val = np.zeros((data_info.train_img_num * extend, data_info.class_num), dtype=np.float32)
    labsBuf = np.zeros((data_info.train_img_num * extend, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT,
                        data_info.class_num),
                       dtype=np.float32)
    k = 0
    avg_p = 1.0 / data_info.class_num
    print 'generator 2D datas no check...'
    for x, y in generator:
        sample_len = len(y)
        for i in range(sample_len):

            if data_info.enhance_enable is True:
                x[i] = enhance_by_random(x=x[i], data_info=data_info)

            ylist = list(y[i])
            index = ylist.index(1.0)
            if data_info.class_name_dic_t[index] == 'bg':  # 抑制背景，效果还不错，也够简单
                labsBuf[k] = avg_p
            else:
                img = np.expand_dims(x[i], axis=0)
                y_p, label_vect = data_info.model.predict(img)
                label = strengthen_method_median_power_tristate(label=label_vect[0], index=index,data_info=data_info)
                labsBuf[k] = label

            x_val[k] = x[i]
            k += 1
            if k == data_info.train_img_num * extend:
                break

        print k
        if k == data_info.train_img_num * extend:
            print '2D datas are....'
            print(type(x_val), x_val.shape, x_val.dtype)
            print(type(y_val), y_val.shape, y_val.dtype)
            print(type(labsBuf), labsBuf.shape, labsBuf.dtype)

            if data_info.label_mode == 2:
                return (x_val, {'main_out': y_val, 'seg_out': labsBuf})
            else:
                return (x_val, labsBuf)


# one_hot train,
# val datas = train datas
def one_hot_getValDatas(data_info=None,err_sample_check=False):
    extend = data_info.val_data_extend
    x_val = np.zeros((data_info.val_img_num * extend, data_info.IMG_ROW, data_info.IMG_COL,3), dtype=np.float32)
    y_val = np.zeros((data_info.val_img_num * extend, data_info.class_num), dtype=np.float32)
    k = 0
    avg_p = 1.0 / data_info.class_num
    print 'generator val datas in one-hot...'
    for x, y in data_info.val_generator:
        sample_len = len(y)

        for i in range(sample_len):

            if data_info.enhance_enable is True:
                x[i] = enhance_by_random(x=x[i], data_info=data_info)

            ylist = list(y[i])
            index = ylist.index(1.0)
            if data_info.class_name_dic_t[index] == 'bg':
                y[i, :] = avg_p

            y_val[k] = y[i]
            x_val[k] = x[i]
            k += 1
            if k == data_info.val_img_num * extend:
                break

        print 'gen val '+str(k)
        #print y_val[k-sample_len:k]
        if k == data_info.val_img_num * extend:
                break

    print 'val datas are....'
    print(type(x_val), x_val.shape, x_val.dtype)
    print(type(y_val), y_val.shape, y_val.dtype)
    return (x_val,y_val)


# 修改train_generator的输出，重新封装为生成器，方便cpu-gpu连续工作
def train_generator_clearbg(data_info=None):
    length = data_info.val_img_num * data_info.val_data_extend
    data_info.one_hot_x_val = np.zeros((length, data_info.IMG_ROW, data_info.IMG_COL, 3), dtype=np.float32)
    data_info.one_hot_y_val = np.zeros((length, data_info.class_num), dtype=np.float32)
    avg_p = 1.0 / data_info.class_num
    k = 0

    while True:

        x, y = data_info.train_generator.next()
        for i in range(len(y)):

            if data_info.enhance_enable is True:
                x[i] = enhance_by_random(x=x[i], data_info=data_info)

            ylist = list(y[i])
            index = ylist.index(1.0)
            if data_info.class_name_dic_t[index] == 'bg':
                y[i, :] = avg_p

            if k < length:
                data_info.one_hot_x_val[k] = x[i]
                data_info.one_hot_y_val[k] = y[i]
                k += 1
            if k == length:
                k += 1
                print(type(data_info.one_hot_x_val), data_info.one_hot_x_val.shape,
                      data_info.one_hot_x_val.dtype, np.min(data_info.one_hot_x_val),
                      np.max(data_info.one_hot_x_val))
        yield (x, y)


def img2laebl(img=None,data_info=None): #by opencv
    img = img[:, :, ::-1]
    img = np.uint8(img)
    # cv2.imshow('img',img)
    label = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', label)

    # cv2.imshow('r', label)
    # print label.shape
    ret, label = cv2.threshold(label, data_info.black_label_threash, 255, cv2.THRESH_BINARY)  # cv2.THRESH_TOZERO
    # cv2.imshow("label", label)
    # cv2.imshow('t', label)
    # print(type(label), label.dtype, np.min(label), np.max(label))
    orign_size_label = label
    label = resetSize(label, data_info.REDUCE_TIME)
    label = np.float32(label)
    label /= 255.
    # cv2.imshow('f', label)
    return label,orign_size_label


def imgOverlayBg(img=None,label2D=None,bg_path=None,data_info=None):
    img = img[:, :, ::-1]
    img = np.uint8(img)

    numfiles = 2
    name_list = list(os.path.join(bg_path, name) for name in os.listdir(bg_path))
    bg_random_file= random.sample(name_list, numfiles)
    bg_random_img=loadImage(url=bg_random_file[0],data_info=data_info)
    bg_random_img = bg_random_img[0]
    bg_random_img = np.uint8(bg_random_img)
    bg_random_img = bg_random_img[:, :, ::-1]
    #cv2.imshow('img',img)
    #cv2.imshow('bg',bg_random_img)
    mask = label2D
    mask_not = cv2.bitwise_not(mask)
    #cv2.imshow('mask',mask)
    #cv2.imshow('masknot',mask_not)
    img[:, :, 0]=cv2.bitwise_and(img[:, :, 0], mask)
    img[:, :, 1]=cv2.bitwise_and(img[:, :, 1], mask)
    img[:, :, 2]=cv2.bitwise_and(img[:, :, 2], mask)

    bg_random_img[:, :, 0]=cv2.bitwise_and(bg_random_img[:, :, 0], mask_not)
    bg_random_img[:, :, 1]=cv2.bitwise_and(bg_random_img[:, :, 1], mask_not)
    bg_random_img[:, :, 2]=cv2.bitwise_and(bg_random_img[:, :, 2], mask_not)
    #cv2.imshow('img_c', img)
    #cv2.imshow('bg_c',bg_random_img)
    bg_img = img + bg_random_img
    #cv2.imshow('bg_img',bg_img)
    #cv2.waitKey(0)
    bg_img = bg_img[:, :, ::-1]
    bg_img = np.float32(bg_img)
    return bg_img


def labels2DShow(img=None, labels=None,data_info=None,show=True,titels=None,x_labels=None,save_path=None):
    '''
    show the 2D output for predict Visualization
    :param url: image path
    :return: plt show result   #plt 和 ax 搞混了，日后做实验
    '''
    img = img.copy()
    img = np.uint8(img)

    plt.figure(figsize=(20, 5))
    plt.suptitle(titels[0],fontsize=20)

    for i in range(data_info.class_num):
        ax = plt.subplot(1, data_info.class_num+1, i+1)
        ax.set_title(x_labels[i],fontsize=12)
        # ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        label = labels[:, :, i]
        #print(type(label), label.dtype, np.min(label), np.max(label))
        ax.imshow(label)
        #plt.subplots_adjust(wspace=1, hspace=1)  # 调整子图间距

    ax = plt.subplot(1, data_info.class_num + 1, data_info.class_num + 1)
    ax.set_title(titels[1])
    ax.imshow(img)
    plt.xlabel(titels[2])

    if save_path is not None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 现在
        plt.savefig(save_path + '/' + nowTime + '.jpg')

    if show is True:
        cv2.imshow('img',img[:, :, ::-1])
        cv2.waitKey()
        plt.show()

    plt.close()


def labels2DShow_boost(img=None, labels=None,seg=None, data_info=None,show=True,titels=None,x_labels=None,save_path=None):
    '''
    show the 2D output for predict Visualization
    :param url: image path
    :return: plt show result   #plt 和 ax 搞混了，日后做实验
    '''
    img = img.copy()
    img = np.uint8(img)

    plt.figure(figsize=(20, 5))
    plt.suptitle(titels[0],fontsize=20)
    n = data_info.class_num + 1

    for i in range(n-1):
        ax = plt.subplot(2, n, i+1)
        ax.set_title(x_labels[i],fontsize=12)
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        label = seg[:, :, i]
        #print(type(label), label.dtype, np.min(label), np.max(label))
        ax.imshow(label)

        ax = plt.subplot(2, n, n + i + 1)
        ax.set_title(x_labels[i], fontsize=12)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        label = labels[:, :, i]
        # print(type(label), label.dtype, np.min(label), np.max(label))
        ax.imshow(label)

    ax = plt.subplot(2, n, n)
    ax.set_title(titels[1])
    ax.imshow(img)
    plt.xlabel(titels[2])

    if save_path is not None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 现在
        plt.savefig(save_path + '/' + nowTime + '.jpg')

    if show is True:
        cv2.imshow('img',img[:, :, ::-1])
        cv2.waitKey()
        plt.show()

    plt.close()


def strengthen_method(label=None,index=0,data_info=None): #label[:,:,index]
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            object = label[i,j,index]
            sum = np.sum(label[i, j])
            if object > (sum - object): # other
                label[i, j, :] = 0.0
                label[i, j, index] = 1.0

    return label


def strengthen_method_extremum(label=None,index=0,data_info=None): #label[:,:,index]
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            object = label[i,j,index]
            a = format(label[i, j], '.1f')
            counts = np.bincount(a)
            mode = np.argmax(counts)
            print mode
            if mode < object:  # min
                label[i, j, :] = 0.0
                label[i, j, index] = 1.0
    return label


def strengthen_method_median(label=None,index=0,data_info=None): #label[:,:,index]
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            object = label[i,j,index]
            median = np.median(label[i, j])
            if object > median * 10:  # 远大于
                label[i, j, :] = 0.0
                label[i, j, index] = 1.0
    return label


def strengthen_method_median_power_tristate_pass(label=None,index=0,data_info=None): #label[:,:,index]
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            object = label[i,j,index]
            median = np.median(label[i, j])
            if object > median * 10 and object > 0.6:  # 远大于中位数
                label[i, j, :] = 0.0
                label[i, j, index] = 1.0
            else:
                label[i, j, :] = 0.5 # 3态0.5 或 2态0.0
    return label

def strengthen_method_median_power_tristate(label=None,index=0,data_info=None): #label[:,:,index]
    avg_p = 1.0 / data_info.class_num
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            object = label[i,j,index]
            if data_info.class_num == 2:
                median = min(label[i, j])
            else:
                median = np.median(label[i, j])

            #if object > avg_p * 2 and median < avg_p / 2:   #永远不改
            if median < avg_p * 1e-4:  # 永远不改

                label[i, j, :] = 0.0
                label[i, j, index] = 1.0
            else:
                #label[i, j, :] = 0.5  # 3态0.5 或 2态0.0
                label[i, j, :] = avg_p

    return label

def strengthen_method_median_power_bool(label=None,index=0,data_info=None): #label[:,:,index]
    for i in range(data_info.IMG_ROW_OUT):
        for j in range(data_info.IMG_COL_OUT):
            object = label[i,j,index]
            if data_info.class_num == 2:
                median = min(label[i, j])
            else:
                median = np.median(label[i, j])
            if object > 0.8 and median < 0.2:
                label[i, j, :] = 0.0
                label[i, j, index] = 1.0
            else:
                label[i, j, :] = 0.0
    return label


# no use
def getVal_2Dlabel(data_info=None,model=None):
    print 'creat val 2Dlabel....'
    x, y = data_info.val_generator.next()
    labsBuf = np.zeros((data_info.val_img_num, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, data_info.class_num),
                       dtype=np.float32)

    for i in range(data_info.val_img_num):
        ylist = list(y[i])
        index = ylist.index(1.0)

        if data_info.class_name_dic_t[index] == 'bg_no':  # 抑制背景，效果还不错，也够简单
            y[i][index] = 0
            continue

        img = np.expand_dims(x[i], axis=0)
        y_p, label_vect = model.predict(img)
        if data_info.step_check:
            print
            print y_p
            if index != 0:
                print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index]
                print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index - 1]
                print 'mean is'
                print np.mean(label_vect[0, :, :, index])
                print np.mean(label_vect[0, :, :, index - 1])
            labels2DShow(img=x[i], labels=label_vect[0], data_info=data_info)

        # label = data_label.strengthen_method0(label=label_vect[0,:,:,index],data_info=data_info)
        # labsBuf[i, :, :, index] = label

        label = strengthen_method(label=label_vect[0], index=index, data_info=data_info)
        labsBuf[i] = label

        # label = becomethin2D(label2D=label_vect[index],thin=2)
        # labsBuf[i, :, :, index] = label_vect[0, :, :, index]


        if data_info.step_check:
            print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index]
            if index != 0:
                print label_vect[0, data_info.IMG_ROW_OUT / 2, :, index - 1]
            labels2DShow(img=x[i], labels=labsBuf[i], data_info=data_info)

    print 'x,y,2dlabel....'
    print(type(x), x.shape, x.dtype, np.min(x), np.max(x))
    print(type(y), y.shape, y.dtype, np.min(y), np.max(y))
    print(type(labsBuf), labsBuf.shape, labsBuf.dtype, np.min(labsBuf), np.max(labsBuf))

    return (x,labsBuf)


def becomethin2D(label2D=None, thin=10):
    _, imgRowsOut, imgColsOut, _ = label2D.shape
    for i in range(imgRowsOut):
        start, end = 0, 0
        for j in range(imgColsOut):
            if label2D[i, j].any() > 0 and start == 0:
                start = j
            if start > 0 and label2D[i, j].any() == 0:
                end = j
                label2D[i, start:start+thin, :] = 0
                if end < thin:
                    end = thin
                label2D[i, end-thin:end, :] = 0
                start = 0
                end = 0
                continue

    for i in range(imgColsOut):
        start, end = 0, 0
        for j in range(imgRowsOut):
            if label2D[j, i].any() > 0 and start == 0:
                start = j
            if start > 0 and label2D[j, i].any() == 0:
                end = j
                label2D[start:start+thin, i, :] = 0
                if end < thin:
                    end = thin
                label2D[end-thin:end, i, :] = 0
                start = 0
                end = 0
                continue
    return label2D


def clearbglabs(y=None, data_info=None):
    for i in range(len(y)):
        ylist = list(y[i])
        index = ylist.index(1.0)
        if data_info.class_name_dic_t[index] == 'bg':
            y[i][index] = 0
    return y


print 'out datalabel'

if __name__ == '__main__':
    pre_data_set(data_set_path='/home/heyude/PycharmProjects/mysite/meal/restaurant/kfc')







########## no use, for reference ####################

# 产生X-Y映射数组
# 遍历.../train/samples里的所有文件
def get_imglab_from_bg(path=None, batch_size=32):

    img_datas = np.zeros((batch_size, imgRows, imgCols, 3), dtype=np.uint8)
    labe_ldatas = np.zeros((batch_size, imgRowsOut, imgColsOut, num_classes), dtype=np.uint8)
    buff_index = 0

    for _, _, files in os.walk(path):
        break
    while True:
        for name in files:
            img_url = path + '/' + name
            img = loadImage(img_url)

            img_datas[buff_index] = img
            buff_index += 1

            if buff_index < batch_size:
                continue
            else:
                yield (img_datas[0:buff_index], labe_ldatas[0:buff_index])
                buff_index = 0


def get2Dlabs(x=None, y=None, batch_size=32,step_check=False,data_info=None,lower=0, upper=0):

    labsBuf = np.zeros((batch_size, data_info.IMG_ROW_OUT, data_info.IMG_COL_OUT, data_info.class_num), dtype=np.uint8)
    for i in range(batch_size):
        ylist = list(y[i])
        index = ylist.index(1.0)
        if data_info.class_name_dic_t[index] == 'bg':
            y[i][index] = 0
            #print 'background found...'
            continue
        #print classNameDic_T[index]
        #print y[i]

        label = clear_plate(x[i], check_step=step_check,lower=lower, upper=upper)

        #label = np.float32(label)
        label /= 255
        labsBuf[i, :, :, index] = label
        if step_check:
            labels2DShow(img=x[i], labels=labsBuf[i])

    return y, labsBuf


def resetSize(img,pyrDownTimes=3):

    #img = cv2.resize(img, (640, 426))
    for i in range(pyrDownTimes):
        img = cv2.pyrDown(img)
    return img

def resetSizeUp(img,pyrUpTimes=3):

    #img = cv2.resize(img, (640, 426))
    for i in range(pyrUpTimes):
        img = cv2.pyrUp(img)
    return img


def get_object_hsv():
    print 'coming opencv'
    import opencv_learn
    print 'out opencv'
    dishes = opencv_learn.dishes
    if os.path.exists(dishes.name + "Hmax.npy"):
        opencv_learn.loadDishWare(dishes)
        #dishes.printInfo()
        dishes_color_lower = dishes.hsvList.getHsvLower()
        dishes_color_upper = dishes.hsvList.getHsvUpper()
        print 'hsv th is'
        print dishes_color_lower, dishes_color_upper
    '''
    white_plate = opencv_learn.DishWare("白色餐盘")
    green_plate = opencv_learn.DishWare("绿色餐盘")
    if os.path.exists(white_plate.name + "Hmax.npy"):
        opencv_learn.loadDishWare(white_plate)
        white_plate.printInfo()
        white_plate_color_lower = white_plate.hsvList.getHsvLower()
        white_plate_color_upper = white_plate.hsvList.getHsvUpper()
    if os.path.exists(green_plate.name + "Hmax.npy"):
        opencv_learn.loadDishWare(green_plate)
        green_plate.printInfo()
        green_plate_color_lower = green_plate.hsvList.getHsvLower()
        green_plate_color_upper = green_plate.hsvList.getHsvUpper()
    '''
    print 'out opencv_t'


def clear_self(plateImg=None,color_lower=None, color_upper=None, step_check=False):
    # 用实体颜色屏蔽掉自己
    HSV = cv2.cvtColor(plateImg, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, color_lower, color_upper)
    mask_not = cv2.bitwise_not(mask)
    objectImg = cv2.bitwise_and(plateImg, plateImg, mask=mask_not)

    if step_check:
        cv2.imshow("in", plateImg)
        cv2.imshow("Mask", mask_not)
        print 'hsv min and max'
        print color_lower
        print color_upper
        cv2.imshow("object", objectImg)
        cv2.waitKey(0)
    return objectImg


def clear_plate(img=None, color_lower=None, color_upper=None,check_step=True,lower=0, upper=0):
    plateImg = img.copy()
    plateImg = plateImg[:, :, ::-1]
    plateImg = np.uint8(plateImg)

    objectImg = clear_self(plateImg, lower, upper, check_step)
    #objectImg = clear_self(objectImg, white_plate_color_lower, white_plate_color_upper, check_step)
    #objectImg = clear_self(objectImg, green_plate_color_lower, green_plate_color_upper, check_step)

    # object to label
    '''
    在缩放时我们推荐使用cv2.INTER_AREA， 在扩展时我们推荐使用v2.INTER_CUBIC（慢)和 v2.INTER_LINEAR。
    默认情况下所有改变图像尺寸大小的操作使用的插值方法都是cv2.INTER_LINEAR
    '''
    #print 'resize'
    #if step_check is False:
    #label = cv2.resize(objectImg, (imgColsOut, imgRowsOut), interpolation=cv2.INTER_AREA)  # row/col exchange
    label = resetSize(objectImg, 3)
    #print label.shape
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label, 96, 255, cv2.THRESH_BINARY)
    # ret, label = cv2.threshold(label, labelThreash, 255, cv2.THRESH_TOZERO)
    #cv2.imshow("label", label)
    return label

def mask3Dlabel(src=None, mask=None):
    print 'in mask'
    print type(src)
    print src.shape
    print type(mask)
    print mask.shape

    mask = np.uint8(mask*255)
    label3D = np.uint8(src.copy())
    print(type(src), src.dtype, np.min(src), np.max(src))
    print(type(label3D), label3D.dtype, np.min(label3D), np.max(label3D))
    #mask2 = cv2.inRange(mask, 1, 1)
    label3D[:, :, 0] = cv2.bitwise_and(label3D[:, :, 0], mask)
    label3D[:, :, 1] = cv2.bitwise_and(label3D[:, :, 1], mask)
    label3D[:, :, 2] = cv2.bitwise_and(label3D[:, :, 2], mask)

    plt.figure(figsize=(20, 5))
    ax = plt.subplot(1, 3, 1)
    ax.set_title('a')
    src /= 255.
    ax.imshow(src)

    ax = plt.subplot(1, 3, 2)
    ax.set_title('b')
    print(type(mask), mask.dtype, np.min(mask), np.max(mask))
    ax.imshow(mask)

    ax = plt.subplot(1, 3, 3)
    ax.set_title('c')
    label3D = becomethin(label3D=label3D, thin=20)
    print(type(label3D), label3D.dtype, np.min(label3D), np.max(label3D))
    ax.imshow(label3D)
    plt.show()
    return label3D


def becomethin(label3D=None, thin=10):

    for i in range(imgRowsOut):
        start, end = 0, 0
        for j in range(imgColsOut):
            if label3D[i, j].any() > 0 and start == 0:
                start = j
            if start > 0 and label3D[i, j].any() == 0:
                end = j
                label3D[i, start:start+thin, :] = 0
                if end < thin:
                    end = thin
                label3D[i, end-thin:end, :] = 0
                start = 0
                end = 0
                continue

    for i in range(imgColsOut):
        start, end = 0, 0
        for j in range(imgRowsOut):
            if label3D[j, i].any() > 0 and start == 0:
                start = j
            if start > 0 and label3D[j, i].any() == 0:
                end = j
                label3D[start:start+thin, i, :] = 0
                if end < thin:
                    end = thin
                label3D[end-thin:end, i, :] = 0
                start = 0
                end = 0
                continue
    return label3D


