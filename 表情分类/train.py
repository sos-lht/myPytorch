from __future__ import print_function,division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net import simpleconv3
import os
import torchvision
import torch.utils.data
from torchvision import transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# 使用tensorboard进行可视化
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')  #创建一个SummaryWriter的实例，默认目录名字为logs

## 训练主函数
def train_model(model, criterion,optimizer,schecduler,num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        for phase in ['train','val']:
            if phase == 'train':
                schecduler.step()
                model.train(True)   #设置为训练模式
            else:
                model.train(False)  #设置为验证模式
            
            running_loss = 0.0  #损失变量
            running_accs = 0.0  #精度变量
            number_batch = 0

            # 从dataloader中获取数据
            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                optimizer.zero_grad()    # 清空梯度,以便在下一个步骤中计算新的梯度。
                outputs = model(inputs)  # 前向运行
                _,preds= torch.max(outputs.data,1)  # 用'_'变量接收最大值的值，用'preds'变量是用来接收类别索引的
                loss = criterion(outputs,labels)  # 计算损失
                if phase == 'train':  # 在训练阶段进行误差反向传播和参数更新
                    loss.backward() # 误差反向传播
                    optimizer.step() # 参数更新
                
                running_loss += loss.data.item()
                running_accs += torch.sum(preds == labels).item()
                number_batch += 1

            # 得到每一个epoch的平均损失与精度
            epoch_loss = running_loss / number_batch
            epoch_acc = running_accs / dataset_sizes[phase]

            # 收集精度和损失用于可视化
            if phase == 'train':
                writer.add_scalar('data/trainloss',epoch_loss,epoch)
                writer.add_scalar('data/trainloss',epoch_acc,epoch)
            else:
                writer.add_scalar('data/valloss',epoch_loss,epoch)
                writer.add_scalar('data/valloss',epoch_acc,epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

    writer.close()
    return model

if __name__ == '__main__':
    
    image_size = 64  # 图像统一缩放大小
    crop_size = 48   # 图像裁剪大小，即训练输入大小
    nclass = 4 # 分类类别数
    model = simpleconv3(nclass) # 创建模型
    data_dir = '/mnt/mydisk2/myPytorch/表情分类/data'  # 数据目录

    # 模型缓存接口
    if not os.path.exists('models'):
        os.mkdir('models')

    # 检查GPU是否可用，如果是使用GPU，否使用CPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    print(model)

    # 创建数据预处理函数，训练预处理包括随机裁剪缩放，随机翻转，归一化，验证预处理包括中心裁剪，归一化
    data_transforms = {
        'train':transforms.Compose([
            transforms.RandomResizedCrop(48),   # 随机裁剪，将图像随机裁剪为指定大小（48 x 48）的正方形。
            transforms.RandomHorizontalFlip(),  # 随机水平翻转，以一定概率（默认为0.5）随机翻转图像。
            transforms.ToTensor(),              # 将 PIL 图像转换为 PyTorch 的 Tensor 格式，将图像数据归一化到 [0, 1] 的范围内。
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # 对归一化后的图像数据进行标准化，使其均值为 0.5，标准差为 0.5。
        ]),
        'val':transforms.Compose([   # 测试集的数据预处理都不能用随机
            transforms.Resize(48),  
            transforms.CenterCrop(48),   # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    }

    # 使用torchvision的dataset_ImageFolder接口读取数据
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                        data_transforms[x]) for x in ['train','val']}

    # 创建数据指针，设置batch大小，shuffle，多进程数量
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=4) for x in ['train','val']}
    
    # 获得数据大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}

    # 优化目标是个交叉熵，优化方法使用带动量项的SGD，学习率迭代策略为step,每隔100个epoch,变为原来的0.1倍
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(),lr=0.1,momentum=0.9)   # 定义优化器，学习率为0.1，动量为0.9
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=100,gamma=0.1)  # optimizer_ft是要进行学习率调整的优化器
                                                                                  # step_size：一个epoch中的迭代次数，表示每隔多少个epoch更新一次学习率
                                                                                  #gamma: 学习率调整的倍率因子，即学习率每次更新后要乘上的系数

    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        schecduler=exp_lr_scheduler,
                        num_epochs=300)
    
    torch.save(model.state_dict(),'/mnt/mydisk2/myPytorch/表情分类/models/model.pt')

    
