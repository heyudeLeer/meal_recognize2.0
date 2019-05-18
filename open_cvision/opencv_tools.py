# -*- coding: utf-8 -*-
print 'in opencv_tools'

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

OPEN_CV_VERSION = cv2.__version__

# 1st ,getContours
def getContours(img):

    # 边缘检测
    edges = getEdges(img,7, 5)
    # 查找轮廓
    return findContours(edges)

def findMaxContour(contours):
    # 找出最大的轮廓
    contoursArcLen = []
    for cnt in contours:
        #contoursArcLen.append(cv2.arcLength(cnt, True))
        hull = cv2.convexHull(cnt)
        contoursArcLen.append(cv2.contourArea(hull))

    # print "max arclen is " +str(max(contoursArcLen))
    maxIndex = contoursArcLen.index(max(contoursArcLen))
    return contours[maxIndex]
    #return max(contoursArcLen)

# 2st,get sth by contour
def getArcLen(contour):
    # 获取周长
    arcLen = cv2.arcLength(contour,True)
    #print "max arclen is " + str(arcLen)
    return arcLen

def getArea(contour):
    #获取面积
    area = cv2.contourArea(contour)
    #print "max area is " + str(area)
    return  area

def getCropImgByContour(contour,img):

    c = contour
    src = img
    # find bounding box coordinates
    y, x, h, w = cv2.boundingRect(c)

    '''
    x -= 10
    y -= 10
    if x<0:
       x=0
    if y<0:
       y=0
    h += 200
    w += 200
    '''
    cropImg = src.copy()[x :x + w , y:y + h]
    #cropImg = src.copy()[x-outter:x+w+outter,y-outter:y+h+outter]
    #cv2.imwrite("cutter" + str(i) + ".png", cropImg)
    #cropImg[inner:inner + w, inner:inner + h, 0] = 0
    #cropImg[(outter+inner):(w+outter-inner), (outter+inner):(h+outter-inner)] = 0
    #cropImg = cv2.resize(cropImg,(200,150))
    #cv2.imwrite("cutter.png", cropImg)
    #cv2.imshow("box", cropImg)
    return cropImg


def getCirCleImgByContour(contour,img):
    cnt = contour
    src = img
    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    cv2.circle(src, center, radius, (0, 255, 0), 2)
    #cv2.imshow("circle", src)
    return src


def getHullImgByContour(contour,img):
    cnt = contour
    src = img
    hull = cv2.convexHull(cnt)
    cv2.drawContours(src, [hull], -1, (255, 0, 0), 2)
    #cv2.imshow("Hull", src)
    #arclen = cv2.arcLength(hull, True)
    #area = cv2.contourArea(hull)
    #print "hull arclen and area"
    #print arclen
    #print area
    return src


def getApproxPolyDpByContour(contour,img):
    drawImg = img
    maxContour = contour
    MaxArcLen = getArcLen(maxContour)
    epsilon = 0.01 * MaxArcLen
    approx = cv2.approxPolyDP(maxContour, epsilon, True)
    cv2.drawContours(drawImg, [approx], -1, (0, 255, 0), 2)
    #cv2.imshow("approxPolyDP", drawImg)
    return drawImg


def getColorContour(img,contour,weight):
    temp = img.copy()
    temp2 = img.copy()
    cv2.drawContours(temp, [contour], -1, (32, 128, 32), -1)
    cv2.imshow("allBip", temp)
    HSV = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
    # H, S, V = cv2.split(HSV)
    # print np.average(H)
    # print np.average(S)
    # print np.average(V)
    # return

    # draw (32, 128, 32)的hsv 60, 191, 128 由于不是方形，手工试错crop得到，注意不能修改，
    LowerContours = np.array([60, 191, 128])
    UpperContours = np.array([61, 192, 129])
    mask = cv2.inRange(HSV, LowerContours, UpperContours)
    cv2.imshow("contourMask", mask)

    # mask_not = cv2.bitwise_not(mask)
    # cv2.imshow("MaskNot", mask_not)

    cv2.drawContours(temp2, [contour], -1, (32, 128, 32), weight)  # hsv 60,191,128
    HSV2 = cv2.cvtColor(temp2, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(HSV2, LowerContours, UpperContours)
    cv2.imshow("realContours-Mask2", mask2)

    # mask2_not = cv2.bitwise_not(mask2)
    # cv2.imshow("MaskNot2", mask2_not)

    # maskUse = cv2.bitwise_or(mask_not,mask2_not)
    maskUse2 = cv2.bitwise_and(mask, mask2)

    myContour = cv2.bitwise_and(img, img, mask=maskUse2)
    #cv2.imshow("myContours", myContour)
    return myContour

# tools

# 获取合适帧尺寸(img)
def resetSize(img,pyrDownTimes=3):

    #img = cv2.resize(img, (640, 426))

    for i in range(pyrDownTimes):
        img = cv2.pyrDown(img)

    #localImshow("src", img)

    return img

def getEdges(img,blurKsize,edgeKsize):
    # 边缘检测
    #edges = cv2.Canny(img, 100, 50)
    edges = strokeEdges(img,blurKsize, edgeKsize)
    #localImshow("edges", edges)
    return edges

def strokeEdges(img,blurKsize=7,edgeKsize=5):
    blurredSrc = cv2.medianBlur(img,blurKsize)
    graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    #cv2.Laplacian(blurredSrc, cv2.CV_8U, blurredSrc, ksize=edgeKsize)
    cv2.imshow("edges",graySrc)
    return  graySrc
    #normalLizeInvAlpha = (255 - graySrc)*1.0/255
    #channels = cv2.split(src)
    #cv2.imshow("b",channels[0])
    #cv2.imshow("g",channels[1])
    #cv2.imshow("r",channels[2])
    #for channel in channels:
    #    channel[:] = channel * normalLizeInvAlpha

    #cv2.imshow("b",channels[0])
    #cv2.imshow("g",channels[1])
    #cv2.imshow("r",channels[2])
    #cv2.merge(channels,graySrc)
    #cv2.imshow("des",graySrc)


    #blurredSrc = cv2.GaussianBlur(graySrc,(17,17),0)
    #gauss = graySrc - blurredSrc
    #cv2.imshow("Gauss",gauss)


def findContours(edges):

    ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    #if edges.size > 100000: #dishs
    #    localImshow("plate", thresh)
    #else:
    #    localImshow("dish", thresh)

    contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL

    if len(contours) == 0:
        #print "do not find contours, please re set getEdges() par!"
        return None

    #print "find "+str(len(contours)) + " contours!"
    return contours




    #cv2.drawContours(src, realContours, -1, (255, 0, 0), 20) #一次全部标出
    #cv2.imshow("realContours", src)
    #return 0
    src = img
    for i in range(len(contours)):
        cnt = contours[maxIndex]

        cv2.drawContours(src, [cnt], -1, (255, 0, 0), 2)
        cv2.imshow("realContours" + str(i), src)
        break

        # find bounding box coordinates
        y, x, h, w = cv2.boundingRect(cnt)
        #draw rectangle
        cv2.rectangle(src, (y, x), (y + h, x + w), (0, 255, 0), 2)
        cv2.imshow("bounding", src)
        #cv2.imwrite("contours" + str(i) + ".png", img)

        # calculate center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        cv2.circle(src, center, radius, (0, 255, 0), 2)
        cv2.imshow("circle", src)

        cv2.drawContours(src, [cnt], -1, (255, 0, 0), 2)
        cv2.imshow("realContours" + str(i), src)
        break
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(src, [approx], -1, (0, 255, 0), 2)
        cv2.imshow("approxPolyDP", src)

        hull = cv2.convexHull(cnt)
        cv2.drawContours(src, [hull], -1, (0, 0, 255), 2)
        cv2.imshow("Hull", src)
        break


    #cv2.imshow("contous", src)


    return  contours

def objectBoolean(img=None):
    c = getContours(img)
    cm = findMaxContour(c)
    x,y,_ = img.shape
    canvas = np.zeros((x, y), dtype=np.uint8)
    cv2.drawContours(image=canvas, contours=cm, contourIdx=-1, color=(255,255,255), thickness=-1)
    #cv2.imshow("allBip", canvas)
    return canvas


def getAreaOneDimension(img=None,th=131,canvas=None,check=False):
    areas=None

    #print(type(img),img.dtype, np.min(img), np.max(img))

    # 边缘检测
    #edges = cv2.Canny(img, 100, 50)
    #blurredSrc = cv2.medianBlur(img, 5)
    #cv2.Laplacian(blurredSrc, cv2.CV_8U, blurredSrc, ksize=5)
    #edges = blurredSrc

    edges = np.uint8(img * 255)

    # get轮廓
    ret, thresh = cv2.threshold(edges, th, 255, cv2.THRESH_BINARY)

    if OPEN_CV_VERSION[0] == str(2):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL RETR_LIST #thresh.copy()
    else:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) == 0:
        return areas, canvas

    #print "find "+str(len(contours)) + " contours!"
    # 获取面积
    areas = []
    for contour in contours:
        #contour = cv2.convexHull(contour)

        area = cv2.contourArea(contour)
        area = round( area, 2)  #1.1修正粒度误差
        areas.append(area)

        if canvas is not None:
            cv2.drawContours(image=canvas, contours=[contour], contourIdx=-1, color=255, thickness=1)

    if check is True:
        n = 4
        plt.figure(figsize=(20, 5))

        ax = plt.subplot(1, n, 1)
        ax.set_title('segImg')
        ax.imshow(img)

        ax = plt.subplot(1, n, 2)
        ax.set_title('edges')
        ax.imshow(edges)

        ax = plt.subplot(1, n, 3)
        ax.set_title('thresh')
        ax.imshow(thresh)

        ax = plt.subplot(1, n, 4)
        ax.set_title('canvas')
        ax.imshow(canvas)

        plt.show()

    return areas,canvas


print 'out opencv_tools'


if __name__ == '__main__':

    img = np.zeros((32,32))
    getAreaOneDimension(img=img)

    exit(0)

    from keras.preprocessing import image

    restaurant_path = '/home/heyude/PycharmProjects/data_set/ibm'
    img_path = restaurant_path + '/predictImg/20180317/UNADJUSTEDNONRAW_thumb_2c30.jpg'

    img = image.load_img(img_path,  target_size=(384, 512))
    # print(type(img), np.min(img), np.max(img))
    # img.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # x = np.array(x) / 255.

    print x.shape
    x = x[0]
    x = x[:, :, ::-1]
    x = np.uint8(x)
    cv2.imshow('test',x)
    label = objectBoolean(img=x)
    cv2.waitKey()

    exit(0)


    c = getContours(x)
    cm = findMaxContour(c)

    crop =  getCropImgByContour(cm,x.copy())
    cv2.imshow('crop', crop)

    cir = getCirCleImgByContour(cm,x.copy())
    cv2.imshow('cir', cir)

    hull = getHullImgByContour(cm, x.copy())
    cv2.imshow('hull', hull)

    appr = getApproxPolyDpByContour(cm, x.copy())
    cv2.imshow('appr', appr)

    color = getColorContour(x.copy(),cm,16)
    cv2.imshow('color', color)

    label = cv2.cvtColor(x.copy(), cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', label)
    ret, label = cv2.threshold(label, 96, 255, cv2.THRESH_BINARY)  # cv2.THRESH_TOZERO
    cv2.imshow('th_label', label)
    cv2.waitKey()