import torch.utils.data as data
import os
import torch
import cv2 as cv

def get_train_path(list_path, file_path):
    image = []
    label = []
    with open(list_path,'r') as lines:
        for line in lines:
            img_pth = os.path.join(file_path,line[:-3])
            image.append(img_pth)
            label.append(line[-2:-1])
    return image, label

file_path = '/mnt/mydisk2/myPytorch/表情分类/表情分类数据'
list_path = '/mnt/mydisk2/myPytorch/表情分类/表情分类数据/lists/train.txt'

img,lbl = get_train_path(list_path,file_path)
print(img)
print(lbl)


#数据集
class expression_dataset(data.Dataset):
    #自定义的参数
    def __init__(self,image,label,transforms=None,debug=False,test=False):
        self.paths = image
        self.labels = label
        self.transforms = transforms
        self.debug = debug
        self.test = test

    #返回图片个数
    def __len__(self):
        return len(self.paths)
    
    #获取每个图片
    def __getitem__(self, item):
        # path
        img_path = self.paths[item]
        # read image
        img = cv.imread(img_path)   #BGR
        # RGB
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        # augmentation
        if self.transforms is not None:
            img = self.transforms(img)
        # read label
        label = self.labels[item]
        # return
        return torch.from_numpy(img).float(), int(label)
        