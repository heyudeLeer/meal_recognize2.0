# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

import socket
import time

#import pyttsx
#print sys.path
#import Image
import datetime

print 'in learn_opencv'
class personalTimer:
    def __init__(self):
        self.ISOformat = "%Y-%m-%d-%H:%M:%S"
        self.Personalformat = "%Y-%m-%d-%H_%M_%S"
        self.startCount = 0

    def getISOtime(self):
        return time.strftime(self.ISOformat, time.localtime(time.time()))

    def getPersonaltime(self):
        return time.strftime(self.Personalformat, time.localtime(time.time()))

    def start(self):
        self.startCount = time.time()

    def getSecondTime(self):
        return time.time() - self.startCount

#尺寸过滤参数
#初值留较大裕量，然后根据print信息更新，使阈值更接近实际情况
class SampleThreshold:
    def __init__(self):
        self.minPlateArcLen = 800
        self.minPlateArea = 30000

        self.min5inchArcLen = 0.278
        self.max5inchArcLen = 0.318

        self.min5inchArea =  0.104
        self.max5inchArea = 0.144

        self.min4inchArcLen = 0.24
        self.max4inchArcLen = 0.33

        self.min4inchArea = 0.05
        self.max4inchArea = 0.11


#——————————————————————————————————————————————————————————————#

PRINT_DEBUG = 1
im_debug = 1
styleDishWareList = []
matchedDishWareList = []

dishWareListById = []
foundDishsId = []



#videoPath = "/Users/heyude/work/PycharmProjects/untitled/images7/"
videoPath = "/Users/heyude/Movies/"

capVideo = "sampleVideo.mov"
capVideoQRcode = "QRcode.mov"
video0 = "2017-4-13 下午4.55 拍摄的影片.mov"
video1 = "2017-3-21 下午3.26 拍摄的影片.mov"
video2 = "2017-3-21 下午3.27 拍摄的影片.mov"
video3 = "2017-3-21 下午3.27 拍摄的影片 #2.mov"
video4 = "2017-3-21 下午3.27 拍摄的影片 #3.mov"
video5 = "2017-3-21 下午3.28 拍摄的影片.mov"

video6 = "2017-3-21 下午10.16 拍摄的影片.mov"
video7 = "2017-3-21 下午10.17 拍摄的影片.mov"
video8 = "2017-3-21 下午10.18 拍摄的影片.mov"
video9 = "2017-3-21 下午10.21 拍摄的影片.mov"
video10 = "2017-3-21 下午10.22 拍摄的影片.mov"
video11 = "2017-3-21 下午10.23 拍摄的影片.mov"
video12 = "2017-3-21 下午10.24 拍摄的影片.mov"
video13 = "2017-3-21 下午10.25 拍摄的影片.mov"
video14 = "2017-3-21 下午10.25 拍摄的影片 #2.mov"
video15 = "2017-4-1 下午2.44 拍摄的影片.mov"
video16 = "2017-4-1 下午2.45 拍摄的影片.mov"
video17 = "2017-4-1 下午11.03 拍摄的影片.mov"
video18 = "2017-4-1 下午11.08 拍摄的影片.mov"

class HSVData:
    def __init__(self):
        self.Hmin = 0
        self.Hmax = 0
        self.Smin = 0
        self.Smax = 0
        self.Vmin = 0
        self.Vmax = 0
        self.Haverage = 0
        self.Saverage = 0
        self.Vaverage = 0

class ColorList:
    def __init__(self):
        self.Hmin = []
        self.Hmax = []
        self.Smin = []
        self.Smax = []
        self.Vmin = []
        self.Vmax = []
        self.Haverage = []
        self.Saverage = []
        self.Vaverage = []

    #def __init__(self):
    #    hsvData = HSVData[]

    def privateAppend(self,color):
        self.Hmin.append(color.Hmin)
        self.Hmax.append(color.Hmax)
        self.Smin.append(color.Smin)
        self.Smax.append(color.Smax)
        self.Vmin.append(color.Vmin)
        self.Vmax.append(color.Vmax)
        self.Haverage.append(color.Haverage)
        self.Saverage.append(color.Saverage)
        self.Vaverage.append(color.Vaverage)
        #self.printInfo()
    def privateremove(self,color):
        self.Hmin.remove(color.Hmin)
        self.Hmax.remove(color.Hmax)
        self.Smin.remove(color.Smin)
        self.Smax.remove(color.Smax)
        self.Vmin.remove(color.Vmin)
        self.Vmax.remove(color.Vmax)
        self.Haverage.remove(color.Haverage)
        self.Saverage.remove(color.Saverage)
        self.Vaverage.remove(color.Vaverage)
    def privateBack(self):
        self.Hmin.pop()
        self.Hmax.pop()
        self.Smin.pop()
        self.Smax.pop()
        self.Vmin.pop()
        self.Vmax.pop()
        self.Haverage.pop()
        self.Saverage.pop()
        self.Vaverage.pop()

    def getHsvLower(self):
        if len(self.Hmax):
            #return (min(self.Hmin),min(self.Smin),min(self.Vmin))
            return (min(self.Haverage),min(self.Saverage),min(self.Vaverage))
            #return ((np.float32(sum(self.Hmin)))/len(self.Hmin), (np.float32(sum(self.Smin)))/len(self.Smin),
            #        (np.float32(sum(self.Vmin)))/len(self.Vmin))

    def getHsvUpper(self):
        if len(self.Hmax):
            #return (max(self.Hmax),max(self.Smax),max(self.Vmax)) #取极值群的极值，变化剧烈，过于灵敏
            return (max(self.Haverage), max(self.Saverage), max(self.Vaverage)) #取平均值的极值，效果较好
            #return ((np.float32(sum(self.Hmax))) / len(self.Hmax), (np.float32(sum(self.Smax))) / len(self.Smax),
            #        (np.float32(sum(self.Vmax))) / len(self.Vmax)) #取极值的平均值，效果不好，来回波动

    def printInfo(self):
        print "hmax:"+str(self.Hmax)
        print "hmin:"+str(self.Hmin)
        print "smax"+str(self.Smax)
        print "smin"+str(self.Smin)
        print "vmax"+str(self.Vmax)
        print "vmin"+str(self.Vmin)
        print "Haverage" + str(self.Haverage)
        print "Saverage" + str(self.Saverage)
        print "Vaverage" + str(self.Vaverage)

class DishWare:
    id = 0
    def __init__(self,name):
        # 命名，并分配ID
        self.name = name
        self.id = DishWare.id
        DishWare.id += 1
        dishWareListById.append(self)

        self.hsvList = ColorList()
        self.hullArea = []
        self.hullArcLen = []

        self.sampleFlag = 0

    def privateBack(self):
        self.hsvList.privateBack()
        if len(self.hullArcLen):
            self.hullArcLen.pop()
            self.hullArea.pop()

    def getMatcheTh(self):

        self.thHullArcLenMin = min(self.hullArcLen)  # average() - t
        self.thHullArcLenMax = max(self.hullArcLen)  # average() + t
        #self.thHullArcLenMin = np.average(self.hullArcLen)  # average() - t
        #self.thHullArcLenMax = np.average(self.hullArcLen)  # average() + t
        self.thHullArcLenAverage = np.average(self.hullArcLen)

        self.thHullAreaMin = min(self.hullArea)
        self.thHullAreaMax = max(self.hullArea)
        #self.thHullAreaMin = np.average(self.hullArea)
        #self.thHullAreaMax = np.average(self.hullArea)
        self.thHullAreaAverage = np.average(self.hullArea)

        if self.id == 0:
            print "plate init............"
            areaDeta = (self.thHullAreaMax-self.thHullAreaMin)/8
            self.thHullAreaMin = self.thHullAreaAverage - areaDeta/2
            self.thHullAreaMax = self.thHullAreaAverage + 4*areaDeta

            arcLenDeta = (self.thHullArcLenMax-self.thHullArcLenMin)/8
            self.thHullArcLenMin = self.thHullArcLenAverage -arcLenDeta/2
            self.thHullArcLenMax = self.thHullArcLenAverage+4*arcLenDeta



        self.thHmin = np.average(self.hsvList.Hmin)
        self.thHmax = np.average(self.hsvList.Hmax)
        self.thSmin = np.average(self.hsvList.Smin)
        self.thSmax = np.average(self.hsvList.Smax)
        self.thVmin = np.average(self.hsvList.Vmin)
        self.thVmax = np.average(self.hsvList.Vmax)
        '''
        self.thHmin = min(self.hsvList.Hmin)
        self.thHmax = max(self.hsvList.Hmax)
        self.thSmin = min(self.hsvList.Smin)
        self.thSmax = max(self.hsvList.Smax)
        self.thVmin = min(self.hsvList.Vmin)
        self.thVmax = max(self.hsvList.Vmax)
        '''
        self.thHaverage = np.average(self.hsvList.Haverage)
        self.thSaverage = np.average(self.hsvList.Saverage)
        self.thVaverage = np.average(self.hsvList.Vaverage)



    def printInfo(self):
        print "dish name is " + self.name
        print "dish id is " +str(self.id)
        print "dish hullArclen is " + str(self.hullArcLen)
        print "dish hullArea is " + str(self.hullArea)

        print "---------hsv info----------"
        self.hsvList.printInfo()

    def printMatcheTh(self):
        #print "thArcLenMin is " +str(self.thArcLenMin)
        #print "thArcLenMax is " + str(self.thArcLenMax)
        #print "thArcLenAverage is " + str(self.thArcLenAverage)

        #print "thAreaMin is " + str(self.thAreaMin)
        #print "thAreaMax is " + str(self.thAreaMax)
        #print "thAreaAverage is " + str(self.thAreaAverage)

        print "thHullArcLenMin is " + str(self.thHullArcLenMin)
        print "thHullArcLenMax is " + str(self.thHullArcLenMax)
        #print "thHullArcLenAverage is " + str(self.thHullArcLenAverage)

        print "thHullAreaMin is " + str(self.thHullAreaMin)
        print "thHullAreaMax is " + str(self.thHullAreaMax)
        #print "thHullAreaAverage is " + str(self.thHullAreaAverage)
'''
        print "thHmin is " + str(self.thHmin)
        print "thHmax is " + str(self.thHmax)
        print "thHaverage is " + str(self.thHaverage)

        print "thSmin is " + str(self.thSmin)
        print "thSmax is " + str(self.thSmax)
        print "thSaverage is " + str(self.thSaverage)

        print "thVmin is " + str(self.thVmin)
        print "thVmax is " + str(self.thVmax)
        print "thVaverage is " + str(self.thVaverage)
'''



# 边缘检测
# self.edges = self.getEdges(7,5)
# 查找轮廓
# self.contour = self.findContours()
# self.arcLen = self.getArcLength()
# self.cropImg = self.getCropImgByContour()
# self.myColorContour = self.getColorContour()


# 按轮廓裁减
#   cutByContours(src,contours,10,20)

# self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
# self.area = cv2.contourArea(o)#oriented

# 获取合适帧尺寸(img)
def resetSize(img,pyrDownTimes=3):

    #img = cv2.resize(img, (640, 426))

    for i in range(pyrDownTimes):
        img = cv2.pyrDown(img)

    #localImshow("src", img)

    return img

def getHsvByHandCrop(img,(x,y,w,h)):

    #手工裁减出特征区域
    crop = img[y:y+h,x:x+w]
    #localImshow("crop", crop)

    hsvData = getHSV(crop)

    return hsvData

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

def localImshow(par1, par2):
    if im_debug:
        cv2.imshow(par1, par2)



def getHSV(img):

    #采集颜色
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(HSV)

    h = bytearray(h)
    s = bytearray(s)
    v = bytearray(v)

    h = deleElement(h)
    s = deleElement(s)
    v = deleElement(v)


    if PRINT_DEBUG:
        print "sample HSV max:"
        print max(h),max(s),max(v)
        print "sample HSV min"
        print min(h),min(s),min(v)
        print "sample HSV average"
        print np.average(h),np.average(s),np.average(v)
        #print np.mean(h),np.mean(s),np.mean(v)
        #print np.median(h),np.median(s),np.median(v)

    Hmax = max(h)
    Hmin = min(h)
    Smax = max(s)
    Smin = min(s)
    Vmax = max(v)
    Vmin = min(v)

    tempColor = HSVData()
    tempColor.Hmax = Hmax
    tempColor.Hmin = Hmin
    tempColor.Smax = Smax
    tempColor.Smin = Smin
    tempColor.Vmax = Vmax
    tempColor.Vmin = Vmin
    tempColor.Haverage = np.average(h)
    tempColor.Saverage = np.average(s)
    tempColor.Vaverage = np.average(v)

    return tempColor

def getObjectByHsv(img,colorLower,colorUpper):
    # 用实体颜色屏蔽掉其他物体
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, colorLower, colorUpper)
    newimg = cv2.bitwise_and(img, img, mask=mask)

    return newimg

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
    #cv2.imshow("lapla",graySrc)
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

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL

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
    return cropImg

def getZbarCropByContour(contour,img):

    c = contour
    src = img
    # find bounding box coordinates
    y, x, h, w = cv2.boundingRect(c)
    #x -= 10
    #y -= 10
    #if x<0:
    #    x=0
    #if y<0:
    #    y=01
    #h += 4
    #w += 4

    cropImg = src.copy()[x :x + w , y:y + h]
    #cropImg = src.copy()[x-outter:x+w+outter,y-outter:y+h+outter]
    #cv2.imwrite("cutter" + str(i) + ".png", cropImg)
    #cropImg[inner:inner + w, inner:inner + h, 0] = 0
    #cropImg[(outter+inner):(w+outter-inner), (outter+inner):(h+outter-inner)] = 0
    #cropImg = cv2.resize(cropImg,(200,150))
    #cv2.imwrite("cutter.png", cropImg)
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
    cv2.imshow("circle", src)

def getHullImgByContour(contour,img):
    cnt = contour
    src = img
    hull = cv2.convexHull(cnt)
    cv2.drawContours(src, [hull], -1, (255, 0, 0), 2)
    cv2.imshow("Hull", src)
    arclen = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    print "hull arclen and area"
    print arclen
    print area
    return hull


def getApproxPolyDpByContour(contour,img):
    drawImg = img.copy
    maxContour = contour
    MaxArcLen = getArcLen(maxContour)
    epsilon = 0.01 * MaxArcLen
    approx = cv2.approxPolyDP(maxContour, epsilon, True)
    cv2.drawContours(drawImg, [approx], -1, (0, 255, 0), 2)
    cv2.imshow("approxPolyDP", drawImg)
    return approx


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

def getPlateContour(img):
    color_lower = plate.hsvList.getHsvLower()
    color_upper = plate.hsvList.getHsvUpper()
    if color_upper == None:
        print "hsv is Null"
        return
    objectImg = getObjectByHsv(img, color_lower, color_upper)
    #localImshow("plateObject", objectImg)
    contours = getContours(objectImg)
    if contours == None:
        print "plate no contours"
        return None
    maxContour = findMaxContour(contours)
    return maxContour
    # 切出盘子
    #dishSCrop = getCropImgByContour(maxContour, img)
    # cv2.imshow("dishs", dishSCrop)
    #return dishSCrop
def getZbarContour(img):
    color_lower = zbar.hsvList.getHsvLower()
    color_upper = zbar.hsvList.getHsvUpper()
    if color_upper == None:
        print "hsv is Null"
        return None
    objectImg = getObjectByHsv(img, color_lower, color_upper)
    #localImshow("zbarObject", objectImg)
    contours = getContours(objectImg)
    if contours == None:
        print "zbar no contours"
        return None
    maxContour = findMaxContour(contours)


    # 画出餐盘,验证是否完整，条件1
    #drawImg = img.copy()
    #cv2.imshow("zbarHull", drawImg)
    return maxContour

def sampleDishsByHandCropFromVideos(video,sampleTh,dish):

    keycode = 0
    adjust = 0
    movingSpeed = 10
    x, y, w, h = [300, 200, 20, 20]

    success, frame = video.read()
    while success :#and cv2.waitKey(4) == -1:

        if keycode == 27:  # 退出
            #saveDishWare(dish)
            break

        if x<0:
            x = 0
        if y<0:
            y = 0
        if w<1:
            w = 1
        if h<1:
            h = 1
        print x, y, w, h
        src = resetSize(frame, 0)
        img = src.copy()

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("crop-for-hsv", img)

        while (True):
            keycode = cv2.waitKey(4)
            if keycode == 0x31:
                success, frame = video.read()
                break
            if keycode == 0x32:                     #手工crop 采集盘子颜色
                newcolor = getHsvByHandCrop(src.copy(),(x, y, w, h))              #盘子颜色采样
                dish.hsvList.privateAppend(newcolor)
                keycode = 0x33

            if keycode == 0x35:                     #hsv倒退
                if len(dish.hsvList.Hmax)>0:
                    #dish.hsvList.privateBack()
                    dish.privateBack()
                    print "-----remove last save------"

            if keycode == 0x33:

                # 利用之前的信息找餐盘
                plateContour = getPlateContour(src.copy())
                plateHull = cv2.convexHull(plateContour)

                # 画出餐盘轮廓,肉眼验证是否完整
                drawImg = src.copy()
                cv2.drawContours(drawImg, [plateContour], -1, (0, 0, 255), 1)
                cv2.drawContours(drawImg, [plateHull], -1, (255, 0, 0), 1)
                cv2.imshow("crop-for-hsv", drawImg)

                # 盘子轮廓找到后，切出盘子，在小盘里找轮廓，避免干扰，并计算轮廓长度面积备用
                dishSCrop = getCropImgByContour(plateContour, src.copy())
                #cv2.imshow("dishs", dishSCrop)

                #用手工crop的hsv信息获取dish轮廓
                color_lower = dish.hsvList.getHsvLower()
                color_upper = dish.hsvList.getHsvUpper()
                if color_upper == None:
                    print "hsv is Null"
                    continue
                #print "************ use dish hsv to get object ***************"
                #print color_lower
                #print color_upper
                objectImg = getObjectByHsv(dishSCrop.copy(),color_lower,color_upper)    #测试是否合适，合适即可保存hsv
                #localImshow("dishObject",objectImg)
                contours = getContours(objectImg)
                if contours == None:
                    continue
                maxContour = findMaxContour(contours)
                hull = cv2.convexHull(maxContour)


                #肉眼观察dish采集是否可保存
                drawImg = dishSCrop.copy()
                cv2.drawContours(drawImg, [maxContour], -1, (0, 0, 255), 1)
                cv2.drawContours(drawImg, [hull], -1, (255, 0, 0), 1)
                cv2.imshow("dishContour", drawImg)

            #决定存储
            if keycode == 0x34:

                hullPlateArcLen = cv2.arcLength(plateHull,True)
                hullPlateArea = cv2.contourArea(plateHull)

                hullArclen = cv2.arcLength(hull, True)
                hullArea = cv2.contourArea(hull)


                hullArcLenScale = hullArclen / hullPlateArcLen
                hullAreaScale = hullArea / hullPlateArea

                dish.hullArcLen.append(hullArcLenScale)
                dish.hullArea.append(hullAreaScale)

                saveDishWare(dish)

            if keycode == 0x38:
                movingSpeed =  movingSpeed*2
                if movingSpeed > 100:
                    movingSpeed = 100
                print "movingSpeed is" +str(movingSpeed)

            if keycode == 0x39:
                movingSpeed =  movingSpeed/2
                if movingSpeed < 1:
                    movingSpeed = 1
                print "movingSpeed is" +str(movingSpeed)


            if keycode == 0x2c:
                print " > > > "
                adjust = 0  # x,y
            if keycode == 0x2e:
                print " < < < "
                adjust = 1  # w,h

            if keycode == 0x6a:  # left,j
                print "left - -  "
                if adjust:
                    w -= movingSpeed
                else:
                    x -= movingSpeed
                break
            if keycode == 0x69:  # up,i
                print "up - - "
                if adjust:
                    h -= movingSpeed
                else:
                    y -= movingSpeed
                break
            if keycode == 0x6c:  # right,l
                print "right  + + "
                if adjust:
                    w += movingSpeed
                else:
                    x += movingSpeed
                break
            if keycode == 0x6b:  # down,k
                print "down + + "
                if adjust:
                    h += movingSpeed
                else:
                    y += movingSpeed
                break
            if keycode == 27:  # 退出
                break


def getPlateByHandCropFromVideos(video, sampleTh, plate):
    keycode = 0
    adjust = 0
    movingSpeed = 10
    x, y, w, h = [300, 200, 20, 20]


    success, frame = video.read()
    while success:  # and cv2.waitKey(4) == -1:

        if keycode == 27:  # 退出
            # saveDishWare(dish)
            break

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if w < 1:
            w = 1
        if h < 1:
            h = 1
        print x, y, w, h
        src = resetSize(frame, 0)
        img = src.copy()

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("crop-for-hsv", img)

        while (True):
            keycode = cv2.waitKey(4)
            if keycode == 0x31:
                success, frame = video.read()
                break
            if keycode == 0x32:  # 手工crop 采集盘子颜色
                newcolor = getHsvByHandCrop(src.copy(), (x, y, w, h))  # 盘子颜色采样
                plate.hsvList.privateAppend(newcolor)
                keycode = 0x33

            if keycode == 0x35:  # 倒退
                if len(plate.hsvList.Hmax) > 0:
                    plate.hsvList.privateBack()
                    #plate.privateBack()
                    print "remove last crop"

            if keycode == 0x33:

                color_lower = plate.hsvList.getHsvLower()
                color_upper = plate.hsvList.getHsvUpper()
                if color_upper == None:
                    print "hsv is Null"
                    break
                objectImg = getObjectByHsv(src.copy(), color_lower, color_upper)  # 测试是否合适，合适即可保存hsv
                #localImshow("object", objectImg)
                contours = getContours(objectImg)
                if contours == None:
                    break
                # 发现contour
                maxContour = findMaxContour(contours)

                #MaxArcLen = getArcLen(maxContour)
                #MaxArea = getArea(maxContour)
                #print "contours arclen and area:"
                #print MaxArcLen
                #print MaxArea

                hull = cv2.convexHull(maxContour)
                #hullArclen = cv2.arcLength(hull, True)
                #hullArea = cv2.contourArea(hull)
                #print "hull arclen and area:"
                #print hullArclen
                #print hullArea

                #肉眼选择
                drawImg = src.copy()
                cv2.drawContours(drawImg, [maxContour], -1, (0, 0, 255), 2)
                cv2.drawContours(drawImg, [hull], -1, (255, 0, 0), 2)
                cv2.imshow("crop-for-hsv", drawImg)

            if keycode == 0x34:

                hullArclen = cv2.arcLength(hull, True)
                hullArea = cv2.contourArea(hull)
                plate.hullArea.append(hullArea)
                plate.hullArcLen.append(hullArclen)

                saveDishWare(plate)


            if keycode == 0x38:
                movingSpeed = movingSpeed * 2
                if movingSpeed > 100:
                    movingSpeed = 100
                print "movingSpeed is" + str(movingSpeed)

            if keycode == 0x39:
                movingSpeed = movingSpeed / 2
                if movingSpeed < 1:
                    movingSpeed = 1
                print "movingSpeed is" + str(movingSpeed)

            if keycode == 0x2c:
                print " > > > "
                adjust = 0  # x,y
            if keycode == 0x2e:
                print " < < < "
                adjust = 1  # w,h

            if keycode == 0x6a:  # left,j
                print "left - -  "
                if adjust:
                    w -= movingSpeed
                else:
                    x -= movingSpeed
                break
            if keycode == 0x69:  # up,i
                print "up - - "
                if adjust:
                    h -= movingSpeed
                else:
                    y -= movingSpeed
                break
            if keycode == 0x6c:  # right,l
                print "right  + + "
                if adjust:
                    w += movingSpeed
                else:
                    x += movingSpeed
                break
            if keycode == 0x6b:  # down,k
                print "down + + "
                if adjust:
                    h += movingSpeed
                else:
                    y += movingSpeed
                break
            if keycode == 27:  # 退出
                break


def findDishsInPlate(plateImg,plateHullArcLen,plateHullArea):
    drawImg = plateImg.copy()
    for styleDishWare in styleDishWareList:
        #styleDishWare = DishWare(styleDishWare)

        #print "checking dish " +styleDishWare.name + " ........."
        color_lower = styleDishWare.hsvList.getHsvLower()
        color_upper = styleDishWare.hsvList.getHsvUpper()
        objectImg = getObjectByHsv(plateImg, color_lower, color_upper)  # 按颜色提取dish，然后计算轮廓

        #cv2.imshow("foud"+str(styleDishWare.id),objectImg)
        contours = getContours(objectImg)
        if contours == None:
            continue

        #localImshow("dish object", objectImg)
        #print "use max contour match"
        cnt = findMaxContour(contours)
        hull = cv2.convexHull(cnt)
        arcLenHull = cv2.arcLength(hull, True)
        areaHull = cv2.contourArea(hull)

        arcLenScale = arcLenHull / plateHullArcLen
        areaScale = areaHull / plateHullArea
        #print "arcLen and area is"
        #print arcLenScale
        #print areaScale
        '''
        if styleDishWare.id == 3:
            cv2.drawContours(objectImg, [hull], -1, (255, 0, 0), 2)
            cv2.imshow("dish" + str(styleDishWare.id), objectImg)
        '''

        if arcLenScale > styleDishWare.thHullArcLenMin and arcLenScale < styleDishWare.thHullArcLenMax and \
           areaScale >styleDishWare.thHullAreaMin and areaScale < styleDishWare.thHullAreaMax:

            #print arcLenScale
            #print areaScale
            #getCirCleImgByContour(cnt, plateImg.copy())
            #getHullImgByContour(cnt,plateImg.copy())

            foundDishsId.append(int(styleDishWare.id))

            cv2.drawContours(drawImg, [hull], -1, (0, 0, 255), 2)  # 画出
            #cv2.drawContours(objectImg, [hull], -1, (255, 0, 0), 2)
            #cv2.imshow("dish" + str(styleDishWare.id), objectImg)
            print "@@@@@@@@@@ found " + styleDishWare.name + "@@@@@@@@@"

            continue

    print "end..."
    cv2.imshow("dishs", drawImg)

def findDishsInPlateByContours(plateImg,plateHullArcLen,plateHullArea):
    drawImg = plateImg.copy()
    outterImg = plateImg.copy()

    # 用实体颜色屏蔽掉自己
    color_lower = plate.hsvList.getHsvLower()
    color_upper = plate.hsvList.getHsvUpper()
    HSV = cv2.cvtColor(plateImg, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, color_lower, color_upper)
    #mask_not = cv2.bitwise_not(mask)
    newimg = cv2.bitwise_and(plateImg, plateImg, mask=mask)
    #localImshow("object", newimg)


    # 边缘检测
    #edges = getEdges(newimg, 5, 5)
    edges = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    # 查找轮廓
    #return findContours(edges)
    #localImshow("edges", edges)
    ret, thresh = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
    #localImshow("plate", thresh)

    # if edges.size > 100000: #dishs
    #    localImshow("plate", thresh)
    # else:
    #    localImshow("dish", thresh)


    image, contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL

    if len(contours) == 0:
        # print "do not find contours, please re set getEdges() par!"
        return None



    # print "find "+str(len(contours)) + " contours!"
    #return contours

    #cv2.drawContours(drawImg, contours, -1, (255, 0, 0), 2) #一次全部标出
    #cv2.imshow("contours", drawImg)
    #return 0
    i=0
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        areaHull = cv2.contourArea(hull)
        arcLenHull = cv2.arcLength(hull,True)

        arcLenScale = arcLenHull / plateHullArcLen
        areaScale = areaHull / plateHullArea

        #if arcLenHull>178 and arcLenHull<841 and areaHull>5000 and areaHull<15000:
        if arcLenScale > dish1.thHullArcLenMin*0.64 and arcLenScale < dish4.thHullArcLenMax*1.28 and \
                        areaScale > dish1.thHullAreaMin*0.64 and areaScale < dish4.thHullAreaMax*1.28:

            i += 1
            print
            print "area info"
            print arcLenHull
            print areaHull

            cv2.drawContours(drawImg, [cnt], -1, (255, 0, 0), 2)

            outter = getColorContour(outterImg, hull, 20)
            cv2.imshow("myContours"+str(i), outter)

    cv2.imshow("realContours", drawImg)

    print "dishs"
    print  i
    return



    #localImshow("dish object", objectImg)
    #print "use max contour match"
    cnt = findMaxContour(contours)
    hull = cv2.convexHull(cnt)
    arcLenHull = cv2.arcLength(hull, True)
    areaHull = cv2.contourArea(hull)

    arcLenScale = arcLenHull / plateHullArcLen
    areaScale = areaHull / plateHullArea
    #print "arcLen and area is"
    #print arcLenScale
    #print areaScale
    '''
    if styleDishWare.id == 3:
        cv2.drawContours(objectImg, [hull], -1, (255, 0, 0), 2)
        cv2.imshow("dish" + str(styleDishWare.id), objectImg)
    '''

    if arcLenScale > styleDishWare.thHullArcLenMin and arcLenScale < styleDishWare.thHullArcLenMax and \
       areaScale >styleDishWare.thHullAreaMin and areaScale < styleDishWare.thHullAreaMax:

        #print arcLenScale
        #print areaScale
        #getCirCleImgByContour(cnt, plateImg.copy())
        #getHullImgByContour(cnt,plateImg.copy())

        foundDishsId.append(int(styleDishWare.id))

        cv2.drawContours(drawImg, [hull], -1, (0, 0, 255), 2)  # 画出
        #cv2.drawContours(objectImg, [hull], -1, (255, 0, 0), 2)
        #cv2.imshow("dish" + str(styleDishWare.id), objectImg)
        print "@@@@@@@@@@ found " + styleDishWare.name + "@@@@@@@@@"



    print "end..."
    cv2.imshow("dishs", drawImg)

def zbarScan(img, plateHullArcLen,plateHullArea):

    maxContour = getZbarContour(img)
    if maxContour==None:
        return

    zbarHull = cv2.convexHull(maxContour)
    zbarHullArcLen = cv2.arcLength(zbarHull, True)/plateHullArcLen
    zbarHullArea = cv2.contourArea(zbarHull)/plateHullArea

    if zbarHullArcLen > zbar.thHullArcLenMin and zbarHullArcLen < zbar.thHullArcLenMax and \
                    zbarHullArea > zbar.thHullAreaMin and zbarHullArea < zbar.thHullAreaMax:  #发现二维码111111111

        #cv2.imshow("QR-codeSrc", img)
        print "found QR code....."
        zbarCrop = getCropImgByContour(maxContour, img)
        #zbarCrop = cv2.imread(videoPath+"zbar1.png")
        cv2.imshow("QR-code", zbarCrop)


        #grayImg = strokeEdges(zbarCrop,1, 9)
        #ret, grayImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY_INV)

        grayImg = cv2.cvtColor(zbarCrop, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("QR-gray", grayImg)

        width, height = grayImg.shape
        #print "zbar info"
        #print width
        #print height
        raw = grayImg.tostring()


        # wrap image data
        image = zbm.Image(width, height, 'Y800', raw)

        # scan the image for barcodes
        scanner.scan(image)

        # extract results
        for symbol in image:
            # do something useful with results
            if symbol:
                print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data

        return zbarCrop

# 找盘子
def findPlate(img):

    color_lower = plate.hsvList.getHsvLower()
    color_upper = plate.hsvList.getHsvUpper()
    objectImg = getObjectByHsv(img, color_lower, color_upper)  # 获取盘子
    # localImshow("object", objectImg)
    contours = getContours(objectImg)
    if contours == None:
        return None

    maxContour = findMaxContour(contours)
    hull = cv2.convexHull(maxContour)
    hullArcLen = cv2.arcLength(hull,True)
    hullArea = cv2.contourArea(hull)

    if hullArea > plate.thHullAreaMin and hullArea < plate.thHullAreaMax and \
       hullArcLen > plate.thHullArcLenMin and hullArcLen < plate.thHullArcLenMax:  # 条件1，盘子大小不合适，丢弃帧
        return hull
    else:
        return None

    # 方法2，轮训而非最大
    for c in range(len(contours)):
        # for maxContour in contours:
        maxContour = contours[-(c + 1)]  # 从外而内
        hull = cv2.convexHull(maxContour)
        hullArcLen = cv2.arcLength(hull, True)
        hullArea = cv2.contourArea(hull)

        if hullArea > plate.thHullAreaMin and hullArea < plate.thHullAreaMax and \
                        hullArcLen > plate.thHullArcLenMin and hullArcLen < plate.thHullArcLenMax:  # 条件1，盘子大小不合适，丢弃帧
            # 找到盘子
            return hull
    return None

def monitorVideo(video):

    nofound = 0
    hadFoundFlag = 0
    success, frame = video.read()
    src = resetSize(frame, 0)
    #cv2.namedWindow("dishs",cv2.WINDOW_NORMAL)
    cv2.imshow("crop-for-hsv", src)

    while True:

        keycode = cv2.waitKey(4)
        if keycode == 27:  # 退出
            break
        if keycode == 0x31:

            success, frame = video.read()
            src = resetSize(frame, 0)
            img = src.copy()
            cv2.imshow("crop-for-hsv", img)

            # 找盘子
            foundPlate = 0
            plateHull = findPlate(src.copy())
            if plateHull != None:
                # 找到盘子
                nofound = 0
                hadFoundFlag = 1

                print
                print "one frame start..."
                # print "#######  found plate ###########"
                # 画餐盘轮廓和hull
                #drawImg = src.copy()
                # cv2.drawContours(drawImg, [maxContour], -1, (0, 0, 255), 2)
                cv2.drawContours(img, [plateHull], -1, (255, 0, 0), 20)
                cv2.imshow("crop-for-hsv", img)
                plateHullArcLen = cv2.arcLength(plateHull, True)
                plateHullArea = cv2.contourArea(plateHull)

                # 切出盘子
                dishSCrop = getCropImgByContour(plateHull, src)
                #dishSCrop = cv2.imread('/Users/heyude/Pictures/zbar1.png')
                #cv2.imshow("dishs", dishSCrop)
                zbarScan(dishSCrop.copy(),plateHullArcLen,plateHullArea)

                # 查询盘子里的碗
                #findDishsInPlate(dishSCrop,plateHullArcLen,plateHullArea)
                findDishsInPlateByContours(dishSCrop, plateHullArcLen, plateHullArea)

            else:
                nofound += 1
                print "----------no plate found -------"
                if (nofound >= 8 and hadFoundFlag==1):
                    nofound = 0
                    hadFoundFlag = 0
                    print list(foundDishsId)
                    L = deleRepeat(foundDishsId)
                    dishsNum = len(L)
                    toYuYin = "发现" + str(dishsNum) + "个碗  "
                    client.send(toYuYin)

                    for id in L:  # foundDishsId:
                        id = int(id)
                        # print "@@@@ found " +str(dishWareListById[id].id)+"号" + dishWareListById[id].name + "@@@@"
                        toYuYinStr = str(dishWareListById[id].id) + "号" + dishWareListById[id].name + ' '
                        client.send(toYuYinStr)
                    foundDishsId[:] = []

def monitorCamera(video):

    nofound = 0
    hadFoundFlag = 0
    success, frame = video.read()
    src = resetSize(frame, 0)
    #cv2.namedWindow("dishs",cv2.WINDOW_NORMAL)
    cv2.imshow("crop-for-hsv", src)


    while success:

        keycode = cv2.waitKey(1)
        if keycode == 27:  # 退出
            break

        success, frame = video.read()
        print
        print "one frame start..."

        src = resetSize(frame, 0)
        cv2.imshow("crop-for-hsv", src)

        # 找盘子
        plateHull = findPlate(src)
        if plateHull != None:
            # 找到盘子
            nofound = 0
            hadFoundFlag = 1
            # print "#######  found plate ###########"
            # 画餐盘轮廓和hull
            img = src.copy()
            # cv2.drawContours(drawImg, [maxContour], -1, (0, 0, 255), 2)
            cv2.drawContours(img, [plateHull], -1, (255, 0, 0), 20)
            cv2.imshow("crop-for-hsv", img)

            plateHullArcLen = cv2.arcLength(plateHull, True)
            plateHullArea = cv2.contourArea(plateHull)

            # 切出盘子
            dishSCrop = getCropImgByContour(plateHull, src)
            #dishSCrop = cv2.imread('/Users/heyude/Pictures/zbar1.png')
            #cv2.imshow("dishs", dishSCrop)

            #识别二维码
            #zbarScan(dishSCrop.copy(),plateHullArcLen,plateHullArea)

            # 查询盘子里的碗
            findDishsInPlate(dishSCrop,plateHullArcLen,plateHullArea)

        else:
            nofound += 1
            print "----------no plate found -------"
            if (nofound >= 8 and hadFoundFlag):
                nofound = 0
                hadFoundFlag = 0
                print "one person disappear..."
                print list(foundDishsId)
                L = deleRepeat(foundDishsId)
                dishsNum = len(L)
                toYuYin = "发现" + str(dishsNum) + "个碗  "
                client.send(toYuYin)

                for id in L:  # foundDishsId:
                    id = int(id)
                    # print "@@@@ found " +str(dishWareListById[id].id)+"号" + dishWareListById[id].name + "@@@@"
                    toYuYinStr = str(dishWareListById[id].id) + "号" + dishWareListById[id].name + ' '
                    client.send(toYuYinStr)
                foundDishsId[:] = []

def matchingDishWare(tempDishWare):
    for styleDishWare in styleDishWareList:
        #print
        #print
        #styleDishWare.getMatcheTh()

        if  tempDishWare.arcLen > styleDishWare.thArcLenMin and                 \
            tempDishWare.arcLen < styleDishWare.thArcLenMax and                 \
            tempDishWare.area > styleDishWare.thAreaMin and                     \
            tempDishWare.area < styleDishWare.thAreaMax and                     \
            tempDishWare.hsvList.Haverage > styleDishWare.thHaverage - 6  and   \
            tempDishWare.hsvList.Haverage < styleDishWare.thHaverage + 6  and   \
            tempDishWare.hsvList.Saverage > styleDishWare.thSaverage - 30 and   \
            tempDishWare.hsvList.Saverage < styleDishWare.thSaverage + 30 and   \
            tempDishWare.hsvList.Vaverage > styleDishWare.thVaverage - 30 and   \
            tempDishWare.hsvList.Vaverage < styleDishWare.thVaverage + 30:
            matchedDishWareList.append(styleDishWare)
            #print " ----------- matching " + styleDishWare.name + " successful ------------"
            #print

def saveDishWare(*dishs):
    for datas in dishs:
        np.save(datas.name + "name", datas.name)
        np.save(datas.name + "id", datas.id)

        np.save(datas.name + "Hmax", datas.hsvList.Hmax)
        np.save(datas.name + "Hmin", datas.hsvList.Hmin)
        np.save(datas.name + "Smax", datas.hsvList.Smax)
        np.save(datas.name + "Smin", datas.hsvList.Smin)
        np.save(datas.name + "Vmax", datas.hsvList.Vmax)
        np.save(datas.name + "Vmin", datas.hsvList.Vmin)
        np.save(datas.name + "Haverage", datas.hsvList.Haverage)
        np.save(datas.name + "Saverage", datas.hsvList.Saverage)
        np.save(datas.name + "Vaverage", datas.hsvList.Vaverage)

        #np.save(datas.name + "arcLen", datas.arcLen)
        #np.save(datas.name + "area", datas.area)

        np.save(datas.name + "hullArcLen", datas.hullArcLen)
        np.save(datas.name + "hullArea", datas.hullArea)

        print
        print(datas.name + " have saved!")
        datas.printInfo()

#load之前存好的模型，节省训练时间
def loadDishWare(*dish):
    for datas in dish:

        datas.id = np.load(datas.name + "id.npy")

        datas.hsvList.Hmax  = list(np.load(datas.name + "Hmax.npy"))
        datas.hsvList.Hmin = list(np.load(datas.name + "Hmin.npy"))
        datas.hsvList.Smax = list(np.load(datas.name + "Smax.npy"))
        datas.hsvList.Smin = list(np.load(datas.name + "Smin.npy"))
        datas.hsvList.Vmax = list(np.load(datas.name + "Vmax.npy"))
        datas.hsvList.Vmin = list(np.load(datas.name + "Vmin.npy"))
        datas.hsvList.Haverage = list(np.load(datas.name + "Haverage.npy"))
        datas.hsvList.Saverage = list(np.load(datas.name + "Saverage.npy"))
        datas.hsvList.Vaverage = list(np.load(datas.name + "Vaverage.npy"))

        datas.hullArcLen = list(np.load(datas.name + "hullArcLen.npy"))
        datas.hullArea = list(np.load(datas.name + "hullArea.npy"))

        print
        print(datas.name + " have loaded!")



def testAutoPar(*args):
    print "len args is " +str(len(args))
    #for par in  args:
    #    print par[1][0]
    for i in range(len(args)):
        print i
        print  args[i]

def deleElement(L,deleK = 3):

    L = filter(lambda x: x != deleK, L)
    #L = filter(None, L) #去0
    return L
    #return list(L)

def deleRepeat(L):
    return  list(set(L))

def writeCameraVideo(camera,name):
    cap = cv2.VideoCapture(camera)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #videoWrite = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)
    videoWrite = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, size)
    success,frame = cap.read()
    print fps
    print size
    while success:
        videoWrite.write(frame)
        success,frame = cap.read()
        cv2.imshow("UsbCamera",frame)
        keycode = cv2.waitKey(30)
        if keycode == 27:
            break

def sampleDishsByHandCropFromVideos(video,sampleTh,dish):

    keycode = 0
    adjust = 0
    movingSpeed = 10
    x, y, w, h = [300, 200, 20, 20]

    success, frame = video.read()
    while success :#and cv2.waitKey(4) == -1:

        if keycode == 27:  # 退出
            #saveDishWare(dish)
            break

        if x<0:
            x = 0
        if y<0:
            y = 0
        if w<1:
            w = 1
        if h<1:
            h = 1
        print x, y, w, h
        src = resetSize(frame, 0)
        img = src.copy()

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("crop-for-hsv", img)

        while (True):
            keycode = cv2.waitKey(4)
            if keycode == 0x31:
                success, frame = video.read()
                break
            if keycode == 0x32:                     #手工crop 采集盘子颜色
                newcolor = getHsvByHandCrop(src.copy(),(x, y, w, h))              #盘子颜色采样
                dish.hsvList.privateAppend(newcolor)
                keycode = 0x33

            if keycode == 0x35:                     #hsv倒退
                if len(dish.hsvList.Hmax)>0:
                    #dish.hsvList.privateBack()
                    dish.privateBack()
                    print "-----remove last save------"

            if keycode == 0x33:

                # 利用之前的信息找餐盘
                plateContour = getPlateContour(src.copy())
                plateHull = cv2.convexHull(plateContour)

                # 画出餐盘轮廓,肉眼验证是否完整
                drawImg = src.copy()
                cv2.drawContours(drawImg, [plateContour], -1, (0, 0, 255), 1)
                cv2.drawContours(drawImg, [plateHull], -1, (255, 0, 0), 1)
                cv2.imshow("crop-for-hsv", drawImg)

                # 盘子轮廓找到后，切出盘子，在小盘里找轮廓，避免干扰，并计算轮廓长度面积备用
                dishSCrop = getCropImgByContour(plateContour, src.copy())
                #cv2.imshow("dishs", dishSCrop)

                #用手工crop的hsv信息获取dish轮廓
                color_lower = dish.hsvList.getHsvLower()
                color_upper = dish.hsvList.getHsvUpper()
                if color_upper == None:
                    print "hsv is Null"
                    continue
                #print "************ use dish hsv to get object ***************"
                #print color_lower
                #print color_upper
                objectImg = getObjectByHsv(dishSCrop.copy(),color_lower,color_upper)    #测试是否合适，合适即可保存hsv
                #localImshow("dishObject",objectImg)
                contours = getContours(objectImg)
                if contours == None:
                    continue
                maxContour = findMaxContour(contours)
                hull = cv2.convexHull(maxContour)


                #肉眼观察dish采集是否可保存
                drawImg = dishSCrop.copy()
                cv2.drawContours(drawImg, [maxContour], -1, (0, 0, 255), 1)
                cv2.drawContours(drawImg, [hull], -1, (255, 0, 0), 1)
                cv2.imshow("dishContour", drawImg)

            #决定存储
            if keycode == 0x34:

                hullPlateArcLen = cv2.arcLength(plateHull,True)
                hullPlateArea = cv2.contourArea(plateHull)

                hullArclen = cv2.arcLength(hull, True)
                hullArea = cv2.contourArea(hull)


                hullArcLenScale = hullArclen / hullPlateArcLen
                hullAreaScale = hullArea / hullPlateArea

                dish.hullArcLen.append(hullArcLenScale)
                dish.hullArea.append(hullAreaScale)

                saveDishWare(dish)

            if keycode == 0x38:
                movingSpeed =  movingSpeed*2
                if movingSpeed > 100:
                    movingSpeed = 100
                print "movingSpeed is" +str(movingSpeed)

            if keycode == 0x39:
                movingSpeed =  movingSpeed/2
                if movingSpeed < 1:
                    movingSpeed = 1
                print "movingSpeed is" +str(movingSpeed)


            if keycode == 0x2c:
                print " > > > "
                adjust = 0  # x,y
            if keycode == 0x2e:
                print " < < < "
                adjust = 1  # w,h

            if keycode == 0x6a:  # left,j
                print "left - -  "
                if adjust:
                    w -= movingSpeed
                else:
                    x -= movingSpeed
                break
            if keycode == 0x69:  # up,i
                print "up - - "
                if adjust:
                    h -= movingSpeed
                else:
                    y -= movingSpeed
                break
            if keycode == 0x6c:  # right,l
                print "right  + + "
                if adjust:
                    w += movingSpeed
                else:
                    x += movingSpeed
                break
            if keycode == 0x6b:  # down,k
                print "down + + "
                if adjust:
                    h += movingSpeed
                else:
                    y += movingSpeed
                break
            if keycode == 27:  # 退出
                break


def getBgByHandCropFromVideos(path=None, sampleTh=None, plate=None):
    keycode = 0
    adjust = 0
    movingSpeed = 10
    x, y, w, h = [300, 200, 32, 32]

    print path
    for _, _, files in os.walk(path):
        break
    leng = len(files)
    index = 0

    img_url = path + '/' + files[0]
    frame = cv2.imread(img_url)
    while True:
        if keycode == 27:  # 退出
            print 'finsh '+plate.name
            break

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if w < 1:
            w = 1
        if h < 1:
            h = 1
        print x, y, w, h

        if plate.name == '白色餐碟':
            src = resetSize(frame, 0)
        else:
            src = resetSize(frame, 0)



        print src.shape
        img = src.copy()

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1) #绿色提示框
        cv2.imshow("crop-for-hsv", img)

        while True:
            keycode = cv2.waitKey(1)
            if keycode == 0x31:
                index += 1
                if index==leng:
                    print '............sample finshes............'
                    keycode=27 # save and exit
                    break

                img_url = path + '/' + files[index]
                frame = cv2.imread(img_url)
                print 'new img...'
                print frame.shape
                break
            if keycode == 0x32:  # 手工crop 采集盘子颜色
                newcolor = getHsvByHandCrop(src.copy(), (x, y, w, h))  # 盘子颜色采样
                plate.hsvList.privateAppend(newcolor)
                keycode = 0x33

            if keycode == 0x35:  # 倒退
                if len(plate.hsvList.Hmax) > 0:
                    plate.hsvList.privateBack()
                    #plate.privateBack()
                    print "remove last crop"

            if keycode == 0x33:

                color_lower = plate.hsvList.getHsvLower()
                color_upper = plate.hsvList.getHsvUpper()
                if color_upper == None:
                    print "hsv is Null"
                    break
                objectImg = getObjectByHsv(src.copy(), color_lower, color_upper)  # 测试是否合适，合适即可保存hsv
                cv2.imshow("objectImg", objectImg)
                continue

                contours = getContours(objectImg)
                if contours == None:
                    break
                # 发现contour
                maxContour = findMaxContour(contours)

                #MaxArcLen = getArcLen(maxContour)
                #MaxArea = getArea(maxContour)
                #print "contours arclen and area:"
                #print MaxArcLen
                #print MaxArea

                hull = cv2.convexHull(maxContour)
                #hullArclen = cv2.arcLength(hull, True)
                #hullArea = cv2.contourArea(hull)
                #print "hull arclen and area:"
                #print hullArclen
                #print hullArea

                #肉眼选择
                drawImg = src.copy()
                cv2.drawContours(drawImg, [maxContour], -1, (0, 0, 255), 2)
                cv2.drawContours(drawImg, [hull], -1, (255, 0, 0), 2)
                cv2.imshow("crop-for-hsv", drawImg)

            if keycode == 0x34:

                #hullArclen = cv2.arcLength(hull, True)
                #hullArea = cv2.contourArea(hull)
                #plate.hullArea.append(hullArea)
                #plate.hullArcLen.append(hullArclen)

                saveDishWare(plate)
                print 'save ' + plate.name


            if keycode == 0x38:
                movingSpeed = movingSpeed * 2
                if movingSpeed > 100:
                    movingSpeed = 100
                print "movingSpeed is" + str(movingSpeed)

            if keycode == 0x39:
                movingSpeed = movingSpeed / 2
                if movingSpeed < 1:
                    movingSpeed = 1
                print "movingSpeed is" + str(movingSpeed)

            if keycode == 0x2c:
                print " > > > "
                adjust = 0  # x,y
            if keycode == 0x2e:
                print " < < < "
                adjust = 1  # w,h

            if keycode == 0x6a:  # left,j
                print "left - -  "
                if adjust:
                    w -= movingSpeed
                else:
                    x -= movingSpeed
                break
            if keycode == 0x69:  # up,i
                print "up - - "
                if adjust:
                    h -= movingSpeed
                else:
                    y -= movingSpeed
                break
            if keycode == 0x6c:  # right,l
                print "right  + + "
                if adjust:
                    w += movingSpeed
                else:
                    x += movingSpeed
                break
            if keycode == 0x6b:  # down,k
                print "down + + "
                if adjust:
                    h += movingSpeed
                else:
                    y += movingSpeed
                break
            if keycode == 27:  # 退出
                break

#dishes = DishWare("白色餐碟")
#dishes = DishWare("白色餐碟2")
#white_plate = DishWare("白色餐盘")
#green_plate = DishWare("绿色餐盘")
def getDishesHsv(name=None):
    print 'opencv hahahahaha'
    dishes = DishWare(name)
    if os.path.exists(dishes.name + "Hmax.npy"):
        loadDishWare(dishes)
        #dishes.printInfo()
        dishes_color_lower = dishes.hsvList.getHsvLower()
        dishes_color_upper = dishes.hsvList.getHsvUpper()
        return dishes_color_lower, dishes_color_upper
    else: print 'NO dishes'


if __name__ == '__main__':
    sampleTh = SampleThreshold()
    if os.path.exists(dishes.name + "Hmax.npy"):
        loadDishWare(dishes)
        #dishes.printInfo()

    '''
    if os.path.exists(white_plate.name + "Hmax.npy"):
        loadDishWare(white_plate)
        white_plate.printInfo()

    if os.path.exists(green_plate.name + "Hmax.npy"):3
        loadDishWare(green_plate)
        green_plate.printInfo()
    '''
    import data_label_o
    getBgByHandCropFromVideos(path=data_label_o.imgPath + 'train/samples2/bg/dishes', plate=dishes)
    getBgByHandCropFromVideos(path=data_label_o.imgPath + 'train/samples2/dishes/方形饼', plate=dishes)
    #getBgByHandCropFromVideos(path=to2D_train.imgPath + 'train/samples2/bg/white_plate', plate=white_plate)
    #getBgByHandCropFromVideos(path=to2D_train.imgPath + 'train/samples2/bg/green_plate', plate=green_plate)
    exit(0)




    #L = [1, 3, 4, 0,0,0,12, 3, 3, 34, 23, 12, 3,0,0,0]
    # ll = deleElement(L) #去0
    #ll = deleElement(L,3)
    #ll = list(set(L))
    #print ll
    #exit(0)

    #data2 = [1,2,3]
    #data3 = [(11,22,333),(44,55,66)]
    #testAutoPar(data2,data3,(1,2))

    #engine = pyttsx.init()
    #engine.say('test')
    #engine.say('蓝色5寸碗')

    #engine.runAndWait()
    #engine.endLoop()
    #print "test "+"语音"

    #engine = pyttsx.init()
   # engine.say('hello world')
   # engine.say('黄色4寸碗')
   # engine.runAndWait()
    # 朗读一次
   # engine.endLoop()

    saveSampleVideo = 0
    if saveSampleVideo:
        writeCameraVideo(1, "sampleVideo.avi")
        exit(0)

    #client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    #client.connect("/tmp/test.sock")

    #instr = "何裕德"
    #client.send(instr)

    #print client.recv(1024)


    # create a reader
    #scanner = zbar.ImageScanner()


    sampleFlag = 1

    sampleTh = SampleThreshold()

    if os.path.exists(plate.name + "Hmax.npy"):
        loadDishWare(plate)
        plate.printInfo()
        plate.getMatcheTh()

    if os.path.exists(zbar.name + "Hmax.npy"):
        loadDishWare(zbar)
        zbar.printInfo()

    dish1 = DishWare("黄色4寸碗")
    if os.path.exists(dish1.name + "Hmax.npy"):
        loadDishWare(dish1)
        dish1.printInfo()

    dish2 = DishWare("黄色6寸碗")
    if os.path.exists(dish2.name + "Hmax.npy"):
        loadDishWare(dish2)
        dish2.printInfo()

    dish3 = DishWare("蓝色4寸碗")
    if os.path.exists(dish3.name + "Hmax.npy"):
        loadDishWare(dish3)
        dish3.printInfo()

    dish4 = DishWare("蓝色6寸碗")
    if os.path.exists(dish4.name + "Hmax.npy"):
        loadDishWare(dish4)
        dish4.printInfo()

    #dish5 = DishWare("黄4黄6碗")
    #if os.path.exists(dish5.name + "Hmax.npy"):
    #    loadDishWare(dish5)
    #    dish5.printInfo()

    print "dishList by id:"
    for dish in dishWareListById:
        print dish.name


###################### 信息采集过程 #########################################
    if sampleFlag == 1:
        #先手动裁减，看看餐盘过滤效果
        #采集餐盘

        print "..........sample plate............"
        video = cv2.VideoCapture(videoPath+capVideo)
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
        print "fps is " + str(fps)
        print "size is " +str(size)
        getPlateByHandCropFromVideos(video,sampleTh,plate)

        print ".......... sample zbar............"
        video = cv2.VideoCapture(videoPath + capVideo)
        sampleDishsByHandCropFromVideos(video, sampleTh, zbar)

        print "..........sample 黄色4寸............"
        video = cv2.VideoCapture(videoPath + capVideo)
        sampleDishsByHandCropFromVideos(video, sampleTh, dish1)

        print "..........sample 蓝色6寸............"
        video = cv2.VideoCapture(videoPath + capVideo)
        sampleDishsByHandCropFromVideos(video, sampleTh, dish4)

        print "..........sample 黄色6寸............"
        video = cv2.VideoCapture(videoPath + capVideo)
        sampleDishsByHandCropFromVideos(video, sampleTh, dish2)

        print "..........sample 蓝色4寸............"
        video = cv2.VideoCapture(videoPath + capVideo)
        sampleDishsByHandCropFromVideos(video, sampleTh, dish3)

        #print "..........sample dish5............"
        #video = cv2.VideoCapture(videoPath + capVideo)
        #sampleDishsByHandCropFromVideos(video, sampleTh, dish5)


    #采集完放入队列备用
    #styleDishWareList.append(plate)
    styleDishWareList.append(dish1)
    styleDishWareList.append(dish2)
    styleDishWareList.append(dish3)
    styleDishWareList.append(dish4)
    #styleDishWareList.append(dish5)

    print
    print "style info is:"
    for style in styleDishWareList:
        style.printInfo()

    '''
    print "plate:"
    print plate.thHullArcLenMin
    print plate.thHullArcLenMax
    print plate.thHullAreaMin
    print plate.thHullAreaMax
    '''
    plate.getMatcheTh()
    zbar.getMatcheTh()
    for style in styleDishWareList:
        print
        print style.name
        style.getMatcheTh()
        style.printMatcheTh()



###################### camera视频匹配过程 #########################################
    if sampleFlag==0:
        print
        print "video matching ................................."

    #video = cv2.VideoCapture(1)
    #video = cv2.VideoCapture(videoPath + video18)
    #fps = video.get(cv2.CAP_PROP_FPS)
    #size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #print "fps is " + str(fps)
    #print size


    #monitorVideo(video)
    #monitorCameraDishCrop(video)
    #cv2.destroyAllWindows()
    #video.release()
    #client.close()
    #exit(0)

    #video = cv2.VideoCapture(videoPath + video0)
    #monitorVideo(video)


        video = cv2.VideoCapture(videoPath + capVideo)
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print "fps is " + str(fps)
        print "size is"
        print size
        monitorVideo(video)

        #二维码
        video = cv2.VideoCapture(videoPath + capVideoQRcode)
        monitorVideo(video)

        #实时检测
        video = cv2.VideoCapture(1)
        monitorCamera(video)

# ----------------------- exit --------------------------- #

    cv2.destroyAllWindows()
    video.release()
    client.close()

print 'out learn_opencv'