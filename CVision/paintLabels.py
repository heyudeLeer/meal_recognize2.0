# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from shutil import copyfile
import opencv_tools as cvt

imgPath = "/Users/heyude/video/mi/imgs/"
paintPath = "/Users/heyude/video/mi/paint/"
labelPath = "/Users/heyude/video/mi/label/"
restaurant = {'name':'爱随食',1:'白米饭', 2:'平菇肉片', 3:'鱼香肉丝',4:'番茄鸡蛋'}

img = None
drawing = False #鼠标按下为真
xs = 0
ys = 0 #start
xl = 0
yl = 0 #last
#待做功能：等到手动输入item的value或者name
labelName = None

def draw_circle(event,x,y,flags,param):
    global drawing,xl,yl,xs,ys

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        xs = xl = x
        ys = yl = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            #cv2.circle(img,(x,y),1,(0,0,255),-1)
            cv2.line(img,(xl,yl),(x,y),(0,0,255),2)
            xl = x
            yl = y
            # undoList.append(img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False

def fillLabel(label=None):
    #edges = cv2.Canny(label, 100, 50)
    #ret, binary = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('binary',binary)
    #cv2.waitKey()
    binary = label
    binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #hull = cv2.convexHull(contours) #用于不闭合的轮廓
    cv2.drawContours(label, contours, -1, 255, -1)

    return label


def getLaelFromPainting(painting=None):
    x,y,z = painting.shape
    label = np.zeros((x, y), dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            if painting[i, j, 0] == 0 and \
                            painting[i, j, 1] == 0 and \
                            painting[i, j, 2] == 255:
                label[i,j] = 255
            else:
                painting[i, j, :] = 0
    #cv2.imshow('label', label)
    #cv2.waitKey(0)

    '''
        for i in range(0,x):
        jl = 0
        pp = 0
        for j in range(1,y):
            if label[i,j]>0 and label[i,j-1] < 0:
                pp += 1
                if pp == 1:
                    jl = j
                if pp == 2:
                    for k in range(jl+1,j):
                        label[i,k] = 255.0
                    pp = 0
    '''

    label=fillLabel(label=label)
    cv2.imshow('label_fill', label)
    #cv2.waitKey(0)

    return label


def getNewPaint(files=None,num=None):
    for index in range(num):
        paint_url = paintPath+files[index]
        if not os.path.exists(paint_url):
            img_url = imgPath + files[index]
            copyfile(img_url, paint_url)
            print 'file index is '+ str(index)
            yield files[index]
    yield None


def handPaintingLabel(imgPath=None, savePath=None):
    src = None
    global img
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    for _, _, files in os.walk(imgPath):
        break
    file_num = len(files)
    index = 0
    print 'file num: ' + str(file_num)

    paint_gen = getNewPaint(files=files, num=file_num)
    paint = paint_gen.next()
    if paint is None:
        print '............sample finshes............'
        return
    print 'painting ' + paint
    (shotname, extension) = os.path.splitext(paint)
    ID = shotname
    src = cv2.imread(paintPath + paint)
    item = 0
    print 'item is '+str(item)
    img = src.copy()

    while (1):

        cv2.imshow('image', img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('l'):
            #cv2.line(img, (xs, ys), (xl, yl), (0, 0, 255), 2) #确保闭合
            label = getLaelFromPainting(painting=img.copy())

        if k == ord('s'):
            cv2.imwrite(paintPath + ID + '_' + str(item) + '.png',label)
            item += 1
            print 'item is ' + str(item)
            img = src.copy()
            cv2.destroyWindow('label_fill')
            cv2.destroyWindow('label')

        if k == ord('k'):
            labelName = restaurant.get('name')+':'+restaurant.get(item+1)

        if k == ord('u'):
            img = src.copy()
        if k == ord('r'):
            item -= 1
            print 'item is ' + str(item)
        if k == ord('t'):
            item += 1
            print 'item is ' + str(item)

        if k == ord('n'):
            paint = paint_gen.next()
            if paint is None:
                print '............sample finshes............'
                return
            print 'painting ' + paint
            (shotname, extension) = os.path.splitext(paint)
            ID = shotname
            src = cv2.imread(paintPath+paint)
            item = 0
            print 'item is ' + str(item)
            img = src.copy()

        elif k == 27:
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    handPaintingLabel(imgPath=imgPath,savePath=paintPath)
