import sys

# 将训练集测试集进行划分，valratio即测试集的比例
def splittrain_val(fileall,valratio=0.1):
    fileids = fileall.split('.')   #按点号进行分割来获取文件名和扩展名
    fileid = fileids[len(fileids)-2] #获取文件名
    f = open(fileall)
    ftrain = open(fileid+"_train.txt",'w')  #创建新的文件对象，训练集
    fval = open(fileid+"_val.txt",'w')
    count = 0
    if valratio == 0 or valratio >= 1:
        valratio = 0.1

    interval = (int)(1.0 / valratio) 

    #如果valratio=0.1,以下代码表示每隔10个值取一个valratio,其他都是interval

    while 1:
        line = f.readline()
        if line:
            count = count + 1
            if count % interval == 0:
                fval.write(line)
            else:
                ftrain.write(line)
        else:
            break

splittrain_val(sys.argv[1],0.1)   #从命令行中获取的第一个参数,即所有样本数据的文件路径