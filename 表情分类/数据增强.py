import os
import cv2 as cv
from PIL import Image
from torchvision import transforms as transforms

outfile = '/mnt/mydisk2/myPytorch/表情分类/数据增强数据'
im = Image.open('/mnt/mydisk2/myPytorch/表情分类/数据增强数据/cat.jpg')
# im = cv.imread('/mnt/mydisk2/myPytorch/表情分类/数据增强数据/cat.jpg')
# im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
print(im)

# 反转
# 水平反转
new_im = transforms.RandomHorizontalFlip(p=1)(im)  # p表示概率，1表示100%反转
new_im.save(os.path.join(outfile,'cat_Hflip.jpg'))
# 垂直反转
new_im = transforms.RandomVerticalFlip(p=1)(im)
new_im.save(os.path.join(outfile,'cat_Vflip.jpg'))

# 旋转
new_im = transforms.RandomRotation(45)(im)  
# 45表示随机旋转的角度，旋转后的图片仍保持原来的大小，黑色部分RGB为(0,0,0)
new_im.save(os.path.join(outfile,'cat_Rota.jpg'))

# 缩放
new_im = transforms.Resize((150,200))(im)
new_im.save(os.path.join(outfile,'cat_Resize.jpg'))

# 裁剪
new_im = transforms.RandomCrop((200,300))(im)  # 随机裁剪出200*300的区域
new_im.save(os.path.join(outfile,'cat_RanCrop.jpg'))
new_im = transforms.CenterCrop((300,300))(im)  # 中心裁剪出300*300的区域
new_im.save(os.path.join(outfile,'cat_CetCrop.jpg'))

# 亮度
new_im = transforms.ColorJitter(brightness=5)(im) #brightness表示从全黑开始的亮度
new_im.save(os.path.join(outfile,'cat_bright.jpg'))

# 对比度
new_im = transforms.ColorJitter(contrast=1)(im)
new_im.save(os.path.join(outfile,'cat_contra.jpg'))

# 饱和度
new_im = transforms.ColorJitter(saturation=0.5)(im)
new_im.save(os.path.join(outfile,'cat_saturat.jpg'))