#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================


import os
import sys
import cv2
txts = open(sys.argv[1]).readlines()
for txt in txts:
    imgpath = txt.split(' ')[0]
    img = cv2.imread(imgpath)
    print imgpath," shape is ",img.shape

