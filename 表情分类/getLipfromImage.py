import cv2 as cv
import dlib
import numpy as np
import sys
import os

# 人脸检测器
cascade_path = '/mnt/mydisk2/myPytorch/表情分类/haarcascade_frontalface_default.xml'
cascade = cv.CascadeClassifier(cascade_path)  
# 关键点检测器
PREDICTOR_PATH = '/mnt/mydisk2/myPytorch/表情分类/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH) 

# 关键点检测
def get_landmarks(im):
    rects = cascade.detectMultiScale(im,1.3,5)
    x,y,w,h = rects[0]
    rect = dlib.rectangle(x,y,x+w,y+h)
    return np.matrix([[p.x,p.y] for p in predictor(im,rect).parts()])

# 图像上显示关键点,利用少量数据用来检测处理结果
def annotate_landmarks(im, landmarks):   #输入图像和一个（68，2）的人脸关键点矩阵
    im = im.copy()
    for idx , point in enumerate(landmarks):
        pos = (point[0,0],point[0,1])
        # 绘制该点序号
        cv.putText(im, str(idx), pos,
        fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
        fontScale=0.4,
        color=(0,0,255))
        # 绘制小圆
        cv.circle(im, pos,5,color=(0,255,255))
    return im   

# 有了人脸区域和关键点，将嘴唇区域裁出
def getlipFromImage(im,landmarks):
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0

    # 获取边界
    for i in range(48,68):
        x = landmarks[i,0]
        y = landmarks[i,1]
        
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    print("xmin",xmin)
    print("xmax",xmax)
    print("ymin",ymin)
    print("ymax",ymax)

    # 嘴唇区域宽度和高度
    roiwith = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax,xmin:xmax,0:3]  # 0:3表示要提取的是图像的三个颜色通道（BGR）


    # 最终截取获得的图片是个正方形的图片
    if roiwith > roiheight:
        dstlen = 1.5*roiwith
    else:
        dstlen = 1.5*roiheight

    # 计算要扩充的像素数
    diff_xlen = dstlen - roiwith
    diff_ylen = dstlen - roiheight

    newx = xmin                    
    newy = ymin
 
    imagerows,imagecols,channel = im.shape     # shape:(height,width,channels)
    if newx >= diff_xlen/2 and newx + roiwith + diff_xlen/2 < imagecols:  # 扩大后左右两边都能保持一定间距
        newx = newx - diff_xlen/2
    elif newx < diff_xlen/2:   # 扩大后超出左边界
        newx = 0
    else:                      # 扩大后超出右边界
        newx = imagecols - dstlen

    if newy >= diff_ylen/2 and newy + roiheight + diff_ylen/2 < imagerows:
        newy = newy - diff_ylen/2
    elif newy < diff_ylen/2:
        newy = 0
    else:
        newy = imagerows - dstlen
    
    roi = im[int(newy):int(newy+dstlen),int(newx):int(newx+dstlen),0:3]
    return roi

# 遍历文件夹,列出指定根目录下的所有文件夹和文件
def listfiles(rootDir):
    # 遍历根目录rootDir,返回一个生成器对象，
    # 该对象生成3个值：1、当前遍历的目录路径2、当前目录下的所有子目录名3、当前目录下的所有文件名
    list_dirs = os.walk(rootDir)    
    for root, dirs, files in list_dirs:
        for d in dirs:
            # 把每个子目录的路径与当前根目录路径合并后打印出来
            print (os.path.join(root,d))
        for f in files:
            fileid = f.split('.')[0]   #文件名
            filetype = f.split('.')[1]  #文件类型
            filepath = os.path.join(root,f)  #根目录+文件名
            try:
                im = cv.imread(filepath,1)  # 1为彩色图像，0为灰度图像
                landmarks = get_landmarks(im)
                show_im = annotate_landmarks(im,landmarks)
                roi = getlipFromImage(im,landmarks)

                #将读入的图像文件名中的后缀改成"_mouth.png"，用于存储截取出的嘴唇区域
                roipath = filepath.replace('.'+filetype,'_mouth.png')

                # 读入的图像文件大小为512x512，则显示标注后的图像show_im，并等待用户按下键盘上的任意键。
                if im.shape[1] == 512 and im.shape[0] == 512:
                    cv.imshow("keypoint",show_im)
                    cv.waitKey(0)

                # 将截取出的嘴唇区域roi保存为新的文件roipath
                cv.imwrite(roipath,roi)
            except:
                print (filepath,"processed error")
                continue

listfiles(sys.argv[1])
# 参数来自命令行参数列表中的第二个参数