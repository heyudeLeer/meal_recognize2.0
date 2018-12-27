#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import cv2
import hashlib
import time
import opencv_learn as cvl

videox = "0711.MP4"
videoPath = "/Users/heyude/video/mi/"
imgPath = "/Users/heyude/video/mi/imgs/"

def handSelectImgFromVideo(video=None,savaPath=None):

    width, height = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    success, frame = video.read()
    frame = frame[height / 4:(height / 4) * 3, width / 4:(width / 4) * 3]
    while success:  # and cv2.waitKey(4) == -1:

        keycode = cv2.waitKey(1)
        cv2.imshow("select sample", frame)

        if keycode == 27:  # 退出
            # saveDishWare(dish)
            break

        if keycode == 0x31:
            success, frame = video.read()
            if success:
                frame = frame[height / 4:(height / 4) * 3, width / 4:(width / 4) * 3]

        if keycode == 0x32:  #
            hashValue = getHashTime()
            cv2.imwrite(savaPath + str(hashValue) + ".jpg", frame)

def getHashTime():
    sha1 = hashlib.sha1()
    time_ms = time.time()
    sha1.update(str(time_ms))
    return sha1.hexdigest()


if __name__ == '__main__':

    print
    print "..........sample............"
    video = cv2.VideoCapture(videoPath + videox)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print "video fps is " + str(fps)
    print "video size is " + str(size)
    handSelectImgFromVideo(video=video,savaPath=imgPath)
