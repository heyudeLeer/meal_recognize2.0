
# encoding: utf-8
print 'in train'

import os
import datetime
import math
import time
import numpy as np
import keras
import multiprocessing
from tensorflow.python.client import device_lib

from keras.utils import multi_gpu_model

from keras.models import load_model,Model
from keras.layers import Dropout, Activation, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization,SeparableConv2D
from keras import backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import data_label
import xception_outsize_change as xception

import sys
print(sys.path)

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def get_gpus_num():
    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    print ('gpu num is '+str(num_gpus))
    if not num_gpus:
        raise ValueError('GPU mode was specified, but no GPUs '
                         'were found. To use CPU')
    return num_gpus


def creatXception(data_info=None,Div=32,upsample=False,train=True,name='header_model'):

    # build the network with ImageNet weights
    inputShape = (data_info.IMG_ROW, data_info.IMG_COL, 3)

    if upsample is True:
        base_model = xception.Xception(input_shape=inputShape,Div=Div)
        if train:
            base_model.load_weights(data_info.base_model_weight_file)
        print 'you chose upsample mode'

        res_out = base_model.get_layer('block14_sepconv1_bn').output
        residual = keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(res_out)
        residual = keras.layers.BatchNormalization()(residual)

        x = base_model.output
        x = keras.layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block15_sepconv1')(x)
        x = keras.layers.BatchNormalization(name='block15_sepconv1_bn')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block15_pool')(x)
        x = keras.layers.add([x, residual], name='block15_add')
        x = keras.layers.Activation('relu', name='block15_sepconv1_act')(x)

        x = keras.layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block15_sepconv2')(x)
        x = keras.layers.BatchNormalization(name='block15_sepconv2_bn')(x)
        x = keras.layers.Activation('relu', name='block15_sepconv2_act')(x)

        x = UpSampling2D((2, 2))(x)
        x = keras.layers.concatenate([x, base_model.output])
        x = SeparableConv2D(1024, (3, 3), use_bias=False, padding='same', name='up_conv0')(x)
        x = BatchNormalization(name='up_conv0_bn')(x)
        x = Activation('relu', name='up_conv0_act')(x)

        if data_info.pixel_level == 3 or data_info.pixel_level == 2 :
            x = UpSampling2D((2, 2))(x)
            x = keras.layers.concatenate([x, base_model.get_layer('block13_sepconv2_act').output])
            x = SeparableConv2D(1024, (3, 3), use_bias=False, padding='same', name='up_conv1')(x)
            x = BatchNormalization(name='up_conv1_bn')(x)
            x = Activation('relu', name='up_conv1_act')(x)
        if data_info.pixel_level == 2 :
            x = UpSampling2D((2, 2))(x)
            x = keras.layers.concatenate([x, base_model.get_layer('block4_sepconv2_act').output])
            x = SeparableConv2D(1024, (3, 3), use_bias=False, padding='same', name='up_conv2')(x)
            x = BatchNormalization(name='up_conv2_bn')(x)
            x = Activation('relu', name='up_conv2_act')(x)

        x = Dropout(0.5)(x)
        x = Conv2D(256, (1, 1), use_bias=False, name='out_conv1')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(256, (1, 1), use_bias=False, name='out_conv2')(x)
        x = Dropout(0.5)(x)

    else:

        base_model = xception.Xception(weights='imagenet', input_shape=inputShape,Div=Div)
        data_info.base_model=base_model
        x = base_model.output

        #x = Dropout(0.5)(x)
        x = Conv2D(256, (1, 1), use_bias=False, name='out_conv1')(x)
        #x = Dropout(0.5)(x)

    header_model = Model(inputs=base_model.input, outputs=x, name=name)

    if train:
        x = header_model.output
        x = Conv2D(data_info.class_num, (1, 1), use_bias=False, name='conv_out')(x)
        seg_output = Activation('softmax', name='seg_out')(x)
        x = GlobalAveragePooling2D()(x)
        main_output = Activation('softmax', name='main_out')(x)
        #with tf.device('/cpu:0'):
        model = Model(inputs=header_model.input, outputs=[main_output,seg_output], name='train_model')
        return model
    else:
        return header_model


def train_model(data_set_path=None, data_info=None):

    if data_info.gpu_num > 1:
        data_info.gpu_num = get_gpus_num()

    #get object contour
    # default one_hot
    pre_train = True
    if pre_train:
        model = creatXception(data_info=data_info,Div=16)
        default_weight_file = data_set_path + '/predictInfo/pixel_level'+str(data_info.pixel_level) + '/one_hot_softmax_Div32' + '.hdf5'
        default_one_hot(data_set_path=data_set_path, model=model, weight_file=default_weight_file, data_info=data_info)
        #model.load_weights(default_weight_file)
        del model

    model = creatXception(data_info=data_info,Div=4)
    if pre_train:
        model.load_weights(default_weight_file)
    one_hot_weight_file = data_set_path + '/predictInfo/pixel_level'+str(data_info.pixel_level) + '/one_hot_softmax' + '.hdf5'
    model = one_hot(data_set_path=data_set_path, model=model, weight_file=one_hot_weight_file, data_info=data_info)
    #model.load_weights(one_hot_weight_file)
    if data_info.one_hot_check is True:
        data_label.one_hot_seg(model=model, data_info=data_info, extend=1, thickness=1e-9,img_num=20)
        #return model

    # boost one_hot
    boost = False
    if boost is True:
        data_info.model = model
        model = creatXception(data_info)
        model.load_weights(weight_file)
        if data_info.boost_self_check is True:
            data_label.boost_seg(model=model, data_info=data_info, generator=data_info.train_generator,img_num=10)

        weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/one_hot_boost.hdf5'
        #model = boost_one_hot(model=model,data_set_path=data_set_path, data_info=data_info,weight_file=weight_file)
        model = simple_boost(model=model,data_set_path=data_set_path, data_info=data_info,weight_file=weight_file)
        #model.load_weights(weight_file)
        if data_info.boost_self_check is True:
            data_label.boost_seg(model=model, data_info=data_info,generator=data_info.train_generator,part=1)
            return model

        del data_info.model

    #U-net-based
    robust = True
    if robust is True:
        # one_hot + up
        data_info.model = model
        data_info.base_model_weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/base_model.hdf5'
        data_info.base_model.save_weights(data_info.base_model_weight_file)
        model = creatXception(data_info=data_info,upsample=True, Div=16, name='header_unet')
        robust_weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/robust.hdf5'

        if data_info.boost_self_check is True:
            data_label.boost_seg(model=data_info.model, data_info=data_info, generator=data_info.train_generator,img_num=20)
        model = u_net_based(model=model,data_set_path=data_set_path, data_info=data_info,weight_file=robust_weight_file)
        #model.load_weights(robust_weight_file)

        if data_info.boost_self_check is True:
            data_label.robust_seg(model=model, data_info=data_info,img_num=20)

        del data_info.model

    print '.....trian finish......'

    return model


class my_call_back(keras.callbacks.Callback):

    def __init__(self,data_info=None):
        self.data_info = data_info
        self.index = 0
        self.err_pixels = 10000
        self.patience=1
        self.weights_file=None
        self.best_weights_file='best_boost_weights.hdf5'


    def on_train_begin(self, logs=None):
        print 'train begin'

    def on_epoch_begin(self, epoch=None, logs=None):
        print 'epoch begin'

    def on_epoch_end(self, epoch=None, logs=None):
        print 'epoch end'

        if epoch == -1: # 实验发现，单独训练big_loss_sample并没有明显效果
            #early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=5e-5)
            len = self.data_info.big_loss_len - 1
            x = self.data_info.big_loss_x[:len]
            y = self.data_info.big_loss_y[:len]
            self.model.fit(x=x,y=y, batch_size=self.data_info.batch_size_GPU, epochs=5, verbose=1,
                           #validation_data=(x,y),callbacks=[early_stopping]
                           )
            self.model.stop_training = False

        #print 'train_abs_loss_sum:' + str(self.data_info.train_abs_loss_sum)
        print 'contour_pixels:' + str(self.data_info.contour_pixels)
        print 'data_info.err_object:' + str(self.data_info.err_object)
        if self.data_info.err_object < self.err_pixels:
            self.patience = 1
            self.data_info.model.save_weights(self.best_weights_file)

            self.index += 1
            self.weights_file = 'boost' + str(self.index) + '.hdf5'
            self.model.save_weights(self.weights_file)
            self.data_info.model.load_weights(self.weights_file)
            print 'update weights...'

            if self.data_info.err_object < 20:
                self.patience = 0
                self.boost_finish = True
            elif self.data_info.err_object > self.err_pixels-10:
                self.patience -= 1
            self.err_pixels = self.data_info.err_object

        else:
            self.patience -= 1

        if self.patience == 0:
            print 'err_pixels no reduce in patiences!'
            self.model.stop_training = True
            self.model.load_weights(self.best_weights_file)
            self.patience = 2

        #self.data_info.train_abs_loss_sum = 0
        self.data_info.contour_pixels = 0
        self.data_info.err_object = 0


def default_one_hot(data_set_path=None,model=None, weight_file=None, data_info=None):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) #'rmsprop' #
    loss = {'main_out': 'categorical_crossentropy'}

    try:
        parallel_model = multi_gpu_model(model, gpus=data_info.gpu_num,cpu_merge=False)
        print("Training using multiple GPUs..")
    except ValueError:
        parallel_model = model
        print("Training using single GPU or CPU..")

    data_info.batch_size_GPU = data_info.batch_size_base * 3 * data_info.gpu_num
    data_label.train_generator_init(data_set_path=data_set_path, data_info=data_info)
    data_gen = data_label.train_generator_clearbg(data_info=data_info)  #data_info.train_generator
    print ('batch_size_GPU is '+str(data_info.batch_size_GPU))

    parallel_model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[keras.metrics.categorical_accuracy]
    )
    parallel_model.summary()
    #workers = multiprocessing.cpu_count()
    parallel_model.fit_generator(
        generator=data_gen,
        epochs=7,
        steps_per_epoch=data_info.steps_per_epoch,
    )
    del data_gen
    model.save_weights(weight_file)

    return model


def one_hot(data_set_path=None,model=None, weight_file=None, data_info=None):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) #'rmsprop' #
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-5, momentum=0.9, nesterov=True)#-4,-6
    # Let's train the model using opt or sgd
    #segModel.compile(optimizer=opt, loss='mse', metrics=['accuracy']) # default 1:1

    # loss1 = keras.losses.mse
    # loss2 = keras.losses.msle #'categorical_crossentropy'
    # loss3 = keras.losses.cosine
    # model.compile(optimizer=opt, loss=loss2, metrics=['accuracy'])

    loss = {'main_out': 'categorical_crossentropy'}

    try:
        parallel_model = multi_gpu_model(model, gpus=data_info.gpu_num,cpu_merge=False)
        print("Training using multiple GPUs..")
    except ValueError:
        parallel_model = model
        print("Training using single GPU or CPU..")

    data_info.batch_size_GPU = data_info.batch_size_base * data_info.gpu_num
    data_label.train_generator_init(data_set_path=data_set_path, data_info=data_info)
    data_gen = data_label.train_generator_clearbg(data_info=data_info)  #data_info.train_generator
    print ('batch_size_GPU is '+str(data_info.batch_size_GPU))

    parallel_model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[keras.metrics.categorical_accuracy]
    )
    parallel_model.summary()
    #workers = multiprocessing.cpu_count()
    parallel_model.fit_generator(
        generator=data_gen,
        epochs=6,
        steps_per_epoch=data_info.steps_per_epoch,
        #workers=workers,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
        #validation_data=data_info.val_datas,#data_info.val_generator,
        #callbacks=[checkpoint, early_stopping],
        #max_queue_size=32,
    )
    del data_gen
    #model.save_weights(weight_file)

    # fine tuning and val
    ### froze cnn ,and re train
    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name == 'block12_add':  # block6_add:GPU8
            break

    #if data_info.epoch > 0:
    #    data_info.val_datas = data_label.one_hot_getValDatas(data_info=data_info) #单独跑一趟，后面的训练速度意外的快很多，不明白为什么
    try:
        parallel_model = multi_gpu_model(model, gpus=data_info.gpu_num,cpu_merge=False)
        print("Training using multiple GPUs..")
        data_info.batch_size_GPU = (data_info.batch_size_base + 1) * data_info.gpu_num
    except ValueError:
        parallel_model = model
        print("Training using single GPU or CPU..")
        data_info.batch_size_GPU = (data_info.batch_size_base + 3)

    data_label.train_generator_init(data_set_path=data_set_path, data_info=data_info)
    data_gen = data_label.train_generator_clearbg(data_info=data_info)  # data_info.train_generator
    print ('batch_size_GPU is '+str(data_info.batch_size_GPU))

    parallel_model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    parallel_model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=3e-5)  # val_loss
    checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2,
                                 save_best_only=True, save_weights_only=True)
    parallel_model.fit_generator(
        generator=data_gen,
        epochs=8,
        steps_per_epoch=data_info.steps_per_epoch,
        #workers=workers,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
        validation_data=(data_info.one_hot_x_val,data_info.one_hot_y_val),  # data_info.val_generator,
        callbacks=[checkpoint, early_stopping],
    )
    del data_gen
    model.save_weights(weight_file)

    return model


def boost_one_hot(data_set_path=None, data_info=None,weight_file=None,model=None,epoch=(10,10)):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    loss = {'seg_out': 'categorical_crossentropy'}
    #data_info.label_mode = 2
    #loss = {'main_out': 'categorical_crossentropy', 'seg_out': 'categorical_crossentropy'}
    #loss_weights = {'main_out': 1.0, 'seg_out': 1.0}
    #early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=3e-5)
    #checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2, save_best_only=True,save_weights_only=True)

    #data_info.start_get_val = True

    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name == 'block6_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 6
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    boost_generator.next()
    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白

    model.compile(
        optimizer=opt,
        loss=loss,
        #loss_weights=loss_weights,
        metrics=[keras.metrics.categorical_accuracy]
    )
    #model.summary()

    epoch_end = my_call_back(data_info=data_info)

    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch/2,
        epochs=epoch[0],
        callbacks=[epoch_end]
    )
    del boost_generator

    if data_info.boost_finish is True:
        model.save_weights(weight_file)
        return model

    # fine tuning and val
    #data_info.enhance_enable = False
    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name =='block12_add':# 'block12_sepconv2_act': # 'block12_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 10
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    #boost_generator.next()

    model.compile(
        optimizer=opt,
        loss=loss,
        #loss_weights=loss_weights,
        metrics=[keras.metrics.categorical_accuracy]
    )
    #model.summary()
    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch/2,
        epochs=epoch[1],
        #validation_data=(data_info.boost_x_val, data_info.boost_label_val),
        #callbacks=[checkpoint, early_stopping]
        callbacks=[epoch_end]
    )
    model.save_weights(weight_file)
    del boost_generator

    return model


def simple_boost(data_set_path=None, data_info=None,weight_file=None,model=None,epoch=(10,10)):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    loss = {'seg_out': 'categorical_crossentropy'}
    #data_info.label_mode = 2
    #loss = {'main_out': 'categorical_crossentropy', 'seg_out': 'categorical_crossentropy'}
    #loss_weights = {'main_out': 1.0, 'seg_out': 1.0}
    #early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=3e-5)
    #checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2, save_best_only=True,save_weights_only=True)

    #data_info.start_get_val = True

    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name == 'block6_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 6
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    boost_generator.next()
    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白

    model.compile(
        optimizer=opt,
        loss=loss,
        #loss_weights=loss_weights,
        metrics=[keras.metrics.categorical_accuracy]
    )
    #model.summary()

    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=1,
    )
    del boost_generator

    if data_info.boost_finish is True:
        model.save_weights(weight_file)
        return model

    # fine tuning and val
    #data_info.enhance_enable = False
    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name =='block12_add':# 'block12_sepconv2_act': # 'block12_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 10
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    #boost_generator.next()

    model.compile(
        optimizer=opt,
        loss=loss,
        #loss_weights=loss_weights,
        metrics=[keras.metrics.categorical_accuracy]
    )
    #model.summary()
    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=1,
        #validation_data=(data_info.boost_x_val, data_info.boost_label_val),
        #callbacks=[checkpoint, early_stopping]
    )
    model.save_weights(weight_file)
    del boost_generator

    return model


def u_net_based(data_set_path=None, data_info=None,weight_file=None,model=None):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    loss = {'seg_out': 'categorical_crossentropy'}

    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白
    try:
        parallel_model = multi_gpu_model(model, gpus=data_info.gpu_num,cpu_merge=False)
        print("Training using multiple GPUs..")
    except ValueError:
        parallel_model = model
        print("Training using single GPU or CPU..")
    data_info.batch_size_GPU = (data_info.batch_size_base+1) * data_info.gpu_num
    data_label.train_generator_init(data_set_path=data_set_path, data_info=data_info)
    print ('batch_size_GPU is ' + str(data_info.batch_size_GPU))
    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)

    parallel_model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    parallel_model.summary()
    parallel_model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=8,
    )
    del boost_generator
    model.save_weights('temp.h5')

    # fine tuning and val
    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name =='block13_add': #''block14_sepconv2_act': block12_add
            break

    try:
        parallel_model = multi_gpu_model(model, gpus=data_info.gpu_num,cpu_merge=False)
        print("Training using multiple GPUs..")
        data_info.batch_size_GPU = (data_info.batch_size_base + 4) * data_info.gpu_num
    except ValueError:
        parallel_model = model
        print("Training using single GPU or CPU..")
        data_info.batch_size_GPU = (data_info.batch_size_base + 8)

    data_label.train_generator_init(data_set_path=data_set_path, data_info=data_info)
    print ('batch_size_GPU is ' + str(data_info.batch_size_GPU))
    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)

    parallel_model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    parallel_model.summary()
    parallel_model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=3,
        #validation_data=(data_info.boost_x_val, data_info.boost_label_val),
        #callbacks=[checkpoint, early_stopping]
    )

    model.save_weights(weight_file)
    del boost_generator

    return model


#Obsolete
def tristate2bool(data_set_path=None, data_info=None,fine_tuning=False):

    print 'in train bool...'
    model = creatXception(data_info)

    #weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/newest_step1.hdf5'
    #weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/newest_step_deeper.hdf5'
    #model.load_weights(weight_file)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3,min_delta=0.0001)
    weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/tristate2bool.hdf5'
    checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2,
                                 save_best_only = True, save_weights_only = True)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    loss = {'seg_out': 'mse'}
    loss_weights = None

    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)
    if data_info.train_img_num * data_info.train_data_extend % data_info.batch_size_GPU == 0:
        steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU
    else:
        steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU + 1
    print 'steps_per_epoch_train is '+str(steps_per_epoch_train)

    bool_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    if data_info.epoch >= 0:
        data_info.val_datas = data_label.predict_2Dlabel_datas(data_info=data_info,
                                                            generator=data_info.val_generator, tristate=False)

    #generator_train.next()
    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白

    model.compile(
        optimizer=opt,
        loss=loss,
        loss_weights=loss_weights,
        metrics=['accuracy']
    )
    print 'bool_train...'
    model.fit_generator(
        generator = bool_generator,
        steps_per_epoch = steps_per_epoch_train,
        epochs = data_info.epoch+30,
        #workers = data_info.cpus,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
        # pickle_safe=True,       #坏事，造成卡顿，程序不走了
        validation_data = data_info.val_datas,
        #validation_steps = steps_per_epoch_val,
        callbacks = [ checkpoint,early_stopping],
    )

    if fine_tuning is True:              # 可能效果不明显，时间很长，在没有时间要求的场合可以打开
        ### froze cnn ,and re train
        for layer in model.layers[:]:
            layer.trainable = False
            print 'bool froze ' + layer.name
            if layer.name == 'block4_pool':
                break

        data_info.batch_size_GPU = 6
        model.load_weights(filepath=weight_file)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.00005)
        data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)
        if data_info.train_img_num * data_info.train_data_extend % data_info.batch_size_GPU == 0:
            steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU
        else:
            steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU + 1
        print 'steps_per_epoch_train is ' + str(steps_per_epoch_train)

        model.compile(
            optimizer=opt,
            loss=loss,
            loss_weights=loss_weights,
            metrics=['accuracy']
        )
        model.summary()
        print 'wasai..'
        model.fit_generator(
            generator=bool_generator,
            steps_per_epoch=steps_per_epoch_train,
            epochs=data_info.epoch+20,
            # workers = data_info.cpus,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
            # pickle_safe=True,       #坏事，造成卡顿，程序不走了
            validation_data=data_info.val_datas,
            # validation_steps = steps_per_epoch_val,
            callbacks=[checkpoint, early_stopping],
        )

    model.load_weights(filepath=weight_file)
    return model
#Obsolete
def boost_one_hot_1by1(data_set_path=None, model=None, weight_file=None, data_info=None):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    #early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=2e-4)
    #checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2, save_best_only=True,
    #                             save_weights_only=True)
    model.compile(
        optimizer=opt,
        loss={'seg_out': 'categorical_crossentropy'},  # 'binary_crossentropy'}, #mse
        loss_weights=None,
        metrics=[keras.metrics.categorical_accuracy]
    )

    loop = 0
    eva_loop = 2
    data_info.model = model

    while data_info.epoch >= 0:

        if data_info.boost_self_check:
            model.load_weights(filepath=weight_file)
            x, y = data_label.boost_seg(data_info=data_info,generator=data_info.train_generator)
        else:
            x, y = data_label.predict_2Dlabel_datas_no_check(data_info=data_info,generator=data_info.train_generator)

        if loop >= eva_loop:
            score_pre = model.evaluate(x=x, y=y, batch_size=data_info.batch_size_GPU,verbose=0)
            print score_pre
        history = model.fit(x=x, y=y, batch_size=data_info.batch_size_GPU, epochs=data_info.train_data_extend,
                            #callbacks=[checkpoint, early_stopping],
                            #validation_data=(x, y))
                            )
        if loop >= eva_loop:
            score = model.evaluate(x=x, y=y, batch_size=data_info.batch_size_GPU,verbose=0)
            print score

        #loss_list = history.history.get('loss')
        #print loss_list
        #print history.history
        if loop >= eva_loop:
            loss_reduce = score_pre[0] - score[0]#min(loss_list)
            print 'loss_reduce: ' + str(loss_reduce)
            if loss_reduce < 4e-3:
                print 'no reduce ...'
                break
            else:
                model.save_weights(weight_file)

        loop += 1
        print 'loop is ' + str(loop)
        if loop >= 10:
            print 'over time,hard to train success,or maybe thickness too small'
            break

    model.load_weights(filepath=weight_file)

    return model


# no use
def train_model_by_predict(data_set_path=None, data_info=None,fine_tuning=True,enlarge=False):

    model = creatXception(data_info)

    #weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/newest_step1.hdf5'
    #model.load_weights(weight_file) #三态

    early_stopping = EarlyStopping(monitor='val_loss', patience=2,min_delta=0.00001)
    weight_file_step2 = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/newest_step2_triple.hdf5'
    checkpoint = ModelCheckpoint(weight_file_step2, monitor='val_loss', verbose=2,
                                 save_best_only = True, save_weights_only = True)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-5, momentum=0.9, nesterov=True)  # -4,-6
    # Let's train the model using opt or sgd
    # segModel.compile(optimizer=opt, loss='mse', metrics=['accuracy']) # default 1:1
    data_info.label_mode = 1
    if data_info.label_mode == 1:
        loss = {'seg_out': 'msle'}
        loss_weights = None
    elif data_info.label_mode == 2:
        loss = {'main_out': 'categorical_crossentropy', 'seg_out': 'msle'}
        loss_weights = {'main_out': 1.0, 'seg_out': 1.0 }

    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)
    if data_info.train_img_num * data_info.train_data_extend % data_info.batch_size_GPU == 0:
        steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU
    else:
        steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU + 1

    generator_train = data_label.one_hot_model_produce_2Dlabel_train(data_info=data_info,
                                                                     generator=data_info.train_generator)

    data_info.val_datas = data_label.one_hot_model_produce_2Dlabel_val(data_info=data_info,
                                                                       generator=data_info.val_generator)

    #generator_train.next()
    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白

    model.compile(
        optimizer=opt,
        loss=loss,
        loss_weights=loss_weights,
        metrics=['accuracy']
    )
    model.summary()
    model.fit_generator(
        generator = generator_train,
        steps_per_epoch = steps_per_epoch_train,
        epochs = data_info.epoch+20,
        workers = data_info.cpus,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
        # pickle_safe=True,       #坏事，造成卡顿，程序不走了
        validation_data = data_info.val_datas,
        #validation_steps = steps_per_epoch_val,
        callbacks = [ checkpoint,early_stopping],
    )
    if fine_tuning is True:

        model.load_weights(filepath=weight_file_step2)
        ### froze deep nn
        for layer in model.layers[:]:
            layer.trainable = False
            print 'froze ' + layer.name
            if layer.name == 'block14_sepconv2_act':
                break

        data_info.batch_size_GPU = data_info.batch_size_GPU * 4
        data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)
        if data_info.train_img_num * data_info.train_data_extend % data_info.batch_size_GPU == 0:
            steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU
        else:
            steps_per_epoch_train = data_info.train_img_num * data_info.train_data_extend / data_info.batch_size_GPU + 1
        generator_train = data_label.one_hot_model_produce_2Dlabel_train(data_info=data_info,
                                                                         generator=data_info.train_generator)

        early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.00001)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )
        model.summary()
        model.fit_generator(
            generator=generator_train,
            steps_per_epoch=steps_per_epoch_train,
            epochs=data_info.epoch+20,
            workers=data_info.cpus,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
            # pickle_safe=True,       #坏事，造成卡顿，程序不走了
            validation_data=data_info.val_datas,
            # validation_steps = steps_per_epoch_val,
            callbacks=[checkpoint, early_stopping],
        )

    if enlarge is True:

        data_info.IMG_ROW = data_info.IMG_ROW * data_info.enlarge_size
        data_info.IMG_COL = data_info.IMG_COL * data_info.enlarge_size
        data_info.IMG_ROW_OUT = data_info.IMG_ROW_OUT * data_info.enlarge_size
        data_info.IMG_COL_OUT = data_info.IMG_COL_OUT * data_info.enlarge_size
        del model
        model = creatXception(data_info)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )
        model.summary()

    print 'trian step2 finish...'
    model.load_weights(filepath=weight_file_step2)
    del data_info.model
    return model


#no use
def create_asyn_model(weights_h5=None):

    model = creatXceptionModel(out_dim=num_classes)
    x = model.output

    aux_output = Activation('sigmoid', name='aux_out')(x)
    segModel = Model(inputs=model.input, outputs=aux_output, name='label2D')

    # 2D to point
    x = GlobalAveragePooling2D()(x)
    main_output = Activation('sigmoid', name='main_out')(x) #测试，和softmax的差别
    avgModel = Model(inputs=model.input, outputs=main_output, name='get_shadow')

    if weights_h5 != None and os.path.exists(weights_h5):
        avgModel.load_weights(weights_h5, by_name=True)
        print('shadowModel have loaded ' + weights_h5)

    return avgModel, segModel

#Obsolete
def trainAsynModel(train=False, epch=1, weights_h5=None):
    global segModel

    avgModel, segModel = create_asyn_model(weights_h5=weights_h5)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(trainModel.layers):
    #    print(i, layer.name)
    # exit(0)
    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    #for layer in shadowModel.layers[:]:
    #    layer.trainable = False
    #rgbTo1_layer = trainModel.get_layer(name='rgb2Atomic')
    #rgbTo1_layer.trainable = False

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-5, momentum=0.9, nesterov=True)#-4,-6
    # Let's train the model using opt or sgd
    avgModel.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    segModel.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    avgModel.summary()

    if train is True:
        for i in range(epch):

            segModel.fit_generator(
                generator=data_label_o.get_imglab_from_bg(path=imgPath + 'train/samples2/dishes/background', batch_size=batchSizeGPU),
                steps_per_epoch=18 / batchSizeGPU,
                epochs=1,
                workers=1,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
                pickle_safe=True,
                max_q_size=1,
                # callbacks=[checkpoint, early_stopping],
            )

            avgModel.fit_generator(
                generator=data_label_o.repack_train_generator(),
                steps_per_epoch=imgsNumTrain / batchSizeGPU,
                epochs=1,
                workers=1,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
                pickle_safe=True,
                max_q_size=1,
                #callbacks=[checkpoint, early_stopping],
            )
            print 'loop is '+str(i)

        #print 'evaluate...'
        #score = segModel.evaluate_generator(validation_generator, steps=math.ceil(imgsNumVal / imgsNumTrain) * valTimes, workers=4)
        #print(score)
        mode = mode+'_ibm_sample2_asyn' +str(data_label_o.imgRowsOut) + '_' + str(data_label_o.imgColsOut)
        time_info = time.strftime('%m-%d_%H:%M', time.localtime(time.time()))
        avgModel.save_weights(mode + '_' + time_info + '.h5')

    return segModel

print 'out train'