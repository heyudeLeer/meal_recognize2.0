
# encoding: utf-8
print 'in train'

import os
import datetime
import math
import time
import numpy as np
import keras

from keras.models import load_model,Model
from keras.layers import Dropout, Activation, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization,SeparableConv2D
from keras import backend as K
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



def creatXception(data_info=None,upsample=False):

    # build the network with ImageNet weights
    inputShape = (data_info.IMG_ROW, data_info.IMG_COL, 3)

    if upsample is True:
        base_model = xception.Xception(include_top=False, input_shape=inputShape,Div=16)
        base_model.load_weights(data_info.base_model_weight_file)
        print 'you chose upsample mode'

        x = base_model.output
        x = UpSampling2D((2, 2))(x)
        x = SeparableConv2D(2048, (3, 3), use_bias=False, padding='same', name='up_conv1')(x)
        x = BatchNormalization(name='up_conv1_bn')(x)
        x = Activation('relu', name='up_conv1_act')(x)
        x = keras.layers.concatenate([x, base_model.get_layer('block13_sepconv2_act').output])
        x = SeparableConv2D(3072, (3, 3), use_bias=False, padding='same', name='up_conv2')(x)
        x = BatchNormalization(name='up_conv2_bn')(x)
        x = Activation('relu', name='up_conv2_act')(x)
    else:

        base_model = xception.Xception(weights='imagenet', include_top=False, input_shape=inputShape,Div=8)
        data_info.base_model=base_model
        x = base_model.output

    x = Dropout(0.5)(x)
    x = Conv2D(256, (1, 1), use_bias=False, name='out_conv1')(x)
    x = Conv2D(256, (1, 1), use_bias=False, name='out_conv2')(x)
    x = Conv2D(data_info.class_num, (1, 1), use_bias=False, name='conv_out')(x)
    seg_output = Activation('softmax', name='seg_out')(x)
    x = GlobalAveragePooling2D()(x)
    main_output = Activation('softmax', name='main_out')(x)

    model = Model(inputs=base_model.input, outputs=[main_output,seg_output], name='one_hot_train_model')

    return model


def train_model(data_set_path=None, data_info=None):

    # one_hot
    model = creatXception(data_info)
    weight_file = data_set_path + '/predictInfo/pixel_level'+str(data_info.pixel_level) + '/one_hot_softmax' + '.hdf5'
    model = one_hot(data_set_path=data_set_path, model=model, weight_file=weight_file, data_info=data_info)
    #model.load_weights(weight_file)
    if data_info.one_hot_check is False:
        data_label.one_hot_self_check(model=model, data_info=data_info, extend=1, thickness=1e-9)

    if data_info.boost_self_check is False:
        data_label.predict_2Dlabel_datas_self_check(model=model, data_info=data_info,
                                                    generator=data_info.train_generator)
    # boost one_hot
    data_info.model = model
    model = creatXception(data_info)
    model.load_weights(weight_file)
    weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/one_hot_boost.hdf5'
    model = boost_one_hot(model=model,data_set_path=data_set_path, data_info=data_info,weight_file=weight_file)
    #model.load_weights(weight_file)
    del data_info.model

    boost_again=False
    if boost_again is True:
        data_info.model = model
        model = creatXception(data_info)
        model.load_weights(weight_file)
        weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/one_hot_boost2.hdf5'
        epoch = (2,1)
        model = boost_one_hot(model=model,data_set_path=data_set_path, data_info=data_info,weight_file=weight_file,epoch=epoch)
        #model.load_weights(weight_file)
        del data_info.model

        if data_info.boost_self_check is True:
            data_label.predict_2Dlabel_datas_self_check(model=model,data_info=data_info, generator=data_info.train_generator)

    #U-net-based
    robust = False
    if robust is True:
        # one_hot + up
        data_info.model = model
        data_info.base_model_weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/base_model.hdf5'
        #data_info.base_model.save_weights(data_info.base_model_weight_file)

        model = creatXception(data_info,upsample=True)
        weight_file = data_set_path + '/predictInfo/pixel_level' + str(data_info.pixel_level) + '/robust.hdf5'
        model = u_net_based(model=model,data_set_path=data_set_path, data_info=data_info,weight_file=weight_file)
        #model.load_weights(weight_file)
        del data_info.model

    print '.....trian finish......'
    return model


def one_hot(data_set_path=None,model=None, weight_file=None, data_info=None):

    early_stopping = EarlyStopping(monitor='val_loss', patience=2,min_delta=5e-5)  # val_loss
    checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2,
                                     save_best_only=True, save_weights_only=True)

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
    data_gen = data_label.train_generator_clearbg(data_info=data_info)  #data_info.train_generator

    model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=data_gen,
        epochs=8,
        steps_per_epoch=data_info.steps_per_epoch,
        #workers=data_info.cpus,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
        #validation_data=data_info.val_datas,#data_info.val_generator,
        #callbacks=[checkpoint, early_stopping],
        #max_queue_size=32,
    )
    del data_gen

    # fine tuning and val
    ### froze cnn ,and re train
    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name == 'block6_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 6
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    data_gen = data_label.train_generator_clearbg(data_info=data_info)  # data_info.train_generator

    #if data_info.epoch > 0:
    #    data_info.val_datas = data_label.one_hot_getValDatas(data_info=data_info) #单独跑一趟，后面的训练速度意外的快很多，不明白为什么

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=data_gen,
        epochs=10,
        steps_per_epoch=data_info.steps_per_epoch,
        # workers=data_info.cpus,  # GPU资源是瓶颈，CPU多核没用，反倒需要打开pickle_safe，使CPU等一下GPU，避免溢出
        validation_data=(data_info.one_hot_x_val,data_info.one_hot_y_val),  # data_info.val_generator,
        callbacks=[checkpoint, early_stopping],
    )
    del data_gen
    model.load_weights(weight_file)

    return model


def boost_one_hot(data_set_path=None, data_info=None,weight_file=None,model=None,epoch=(3,2)):

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #
    loss = {'seg_out': 'categorical_crossentropy'}
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=3e-5)
    checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2, save_best_only=True,save_weights_only=True)

    #data_info.start_get_val = True

    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    boost_generator.next()
    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白

    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name == 'block6_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 6
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=epoch[0],
    )
    del boost_generator

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
        metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=epoch[1],
        #validation_data=(data_info.boost_x_val, data_info.boost_label_val),
        #callbacks=[checkpoint, early_stopping]
    )
    model.save_weights(weight_file)
    del boost_generator

    return model


def u_net_based(data_set_path=None, data_info=None,weight_file=None,model=None):

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=5e-5)  # val_loss
    checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=2,
                                 save_best_only=True, save_weights_only=True)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 'rmsprop' #

    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name == 'block10_add':  # block6_add:GPU8
            break
    data_info.batch_size_GPU = 8
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    loss = {'main_out': 'categorical_crossentropy'}
    data_gen = data_label.train_generator_clearbg(data_info=data_info)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=data_gen,
        epochs=3,
        steps_per_epoch=data_info.steps_per_epoch,
    )
    del data_gen

    loss = {'seg_out': 'categorical_crossentropy'}
    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    boost_generator.next()
    # 偶尔必须空跑一次，否则 ValueError: Tensor Tensor('main_out/Softmax:0', shape=(?, 24), dtype=float32) is not an element of this graph.
    # 很重要，但不明白

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=3,
    )
    del boost_generator

    # fine tuning and val
    for layer in model.layers[:]:
        layer.trainable = False
        print 'boost froze' + layer.name
        if layer.name =='block14_sepconv2_act':
            break
    data_info.batch_size_GPU = 12
    data_label.get_data_info(data_set_path=data_set_path, data_info=data_info)

    boost_generator = data_label.predict_2Dlabel_generator(data_info=data_info)
    #boost_generator.next()

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy]
    )
    model.summary()
    model.fit_generator(
        generator=boost_generator,
        steps_per_epoch=data_info.steps_per_epoch,
        epochs=2,
        #validation_data=(data_info.boost_x_val, data_info.boost_label_val),
        #callbacks=[checkpoint, early_stopping]
    )
    #new_weights = model.get_weights()
    #data_info.model.set_weights(new_weights)  # update: model1 load model2 weights

    #nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 现在
    #temp_weight_file = data_set_path + '/predictInfo/pixel_level' + str(
    #    data_info.pixel_level) + '/one_hot_boost-' + nowTime + '.hdf5'
    #model.save_weights(temp_weight_file)

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
            x, y = data_label.predict_2Dlabel_datas_self_check(data_info=data_info,generator=data_info.train_generator)
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