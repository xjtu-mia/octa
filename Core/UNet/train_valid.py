# Input file Data train
# Output file Checkpoint Train_log
# train2是对valid可视化， 这样可以减小内存消耗，增大batch。但是出现了随机变化， 这个问题需要查找
import os
import numpy as np
import time
from PIL import Image
from tqdm import tqdm

import torch
import torchsummary
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import segmentation_models_pytorch as smp
from tensorboardX import SummaryWriter
import sys
sys.path.append(sys.path[0]+'/Core/UNet')
from Model.Data_Loader import Images_Dataset
from Model.Unet_Family import U_Net
from Model.Make_floder import makefloder
from Model.Losses import calc_loss, calc_dice
import copy
import argparse
# GPU 检查GPU是否可用


def main(args):
    train_on_gpu = torch.cuda.is_available()
    if args.gpu < 0 or not train_on_gpu:
        print('Cuda is not available. Training on CPU')
        device = torch.device("cpu")
    else:
        print('Cuda is available. Training on GPU')
        device = torch.device("cuda:" + str(args.gpu))
    # writer = SummaryWriter('./train_log')
    Data_path = args.Data_path
    result_path = args.cnn_Result_path
    makefloder(result_path)
    Batch_Size = args.cnn_batch_size
    Epoch = args.cnn_epochs
    initial_learning_rate = args.cnn_lr
    Num_Workers = args.num_workers  # 使用线程数目,负责并发
    pin_memory = args.pin_memory  # Dataloader 参数. 如果GPU可以使用,则设置为True.
    if train_on_gpu:
        pin_memory = True  # If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.
    valid_loss_min = np.inf
    """　参数选择与设置　"""
    # U-Net 模型选择与设置(Input_channel and Output_channel)
    '''deeplabV3pp = smp.DeepLabV3Plus
    Model = {'deeplabV3pp': deeplabV3pp}'''
    In_ch = 4
    Out_ch = 3
    # 第一次的网络中设置了最终的激活函数是softmax，这里参考之前把这个取消掉
    '''model = Model['deeplabV3pp'](in_channels=In_ch, 
                                classes=Out_ch,
                                activation=None,
                                encoder_weights=None,
                                )#layer - 160
    '''
    model = U_Net(in_ch=In_ch, out_ch=Out_ch)
    model.to(device)

    torchsummary.summary(model, (4, 320, 320))

    # 训练数据集路径
    train_path = os.path.join(Data_path, 'train')
    train_label_path = os.path.join(Data_path, 'train_GT')
    valid_path = os.path.join(Data_path, 'test')
    valid_label_path = os.path.join(Data_path, 'valid_GT')
    # 结果路径设置

    """ 数据加载　"""
    # Dataset 将训练数据集加载到tensor
    Train_data = Images_Dataset(train_dir=train_path, label_dir=train_label_path)
    Valid_data = Images_Dataset(train_dir=valid_path, label_dir=valid_label_path)
    print(f'Train data number is {len(Train_data)};')
    print(f'Train images number is {len(Train_data.train_A_images)};')
    print(f'Label images number is {len(Train_data.label_images)};')

    # 分割 Training 与 Validation 集合

    train_idx = list(range(len(Train_data)))
    valid_idx = list(range(len(Valid_data)))
    train_sampler = SubsetRandomSampler(train_idx)  # 无放回地按照给定的索引列表采样样本元素。
    valid_sampler = SubsetRandomSampler(valid_idx)

    # DataLoader 按批加载数据
    train_loader = torch.utils.data.DataLoader(Train_data, batch_size=Batch_Size, sampler=train_sampler,
                                               num_workers=Num_Workers,
                                               pin_memory=pin_memory)  # 注销了线程，不知道是不是版本问题
    valid_loader = torch.utils.data.DataLoader(Valid_data, batch_size=Batch_Size, sampler=valid_sampler,
                                               num_workers=Num_Workers,
                                               pin_memory=pin_memory)
    valid_loader = copy.deepcopy(valid_loader)
    """ 训练模型的配置（损失函数和优化器）　"""
    # Optimizer 优化器
    opt = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)

    # PyTorch 在torch.optim.lr_scheduler包中提供了一些调整学习率的技术

    T_Max = 60  # 原程序中设置是不正确的
    eta_min = 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_Max, eta_min)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, verbose=True)
    # Loss  损失函数
    loss = calc_loss
    Train_loss = np.inf
    Valid_loss = np.inf

    log = {'train_loss': [], 'valid_loss': [], 'dice_values': [], 'lr': []}
    """ Train """
    for i in range(Epoch):

        train_loss = 0.0
        valid_loss = 0.0
        dice = 0
        start = time.time()
        # lr = scheduler.get_lr()
        # 训练
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            output = model(x)
            lossT = loss(output, y)
            train_loss += lossT.item() * x.size(0)
            lossT.backward()
            opt.step()
        Train_loss = train_loss / len(train_idx)
        print('Epoch: {}/{} \t Learning Rate: {:.3f} \t Training Loss: {:.6f} \t'
              .format(i + 1, Epoch, opt.param_groups[0]['lr'], Train_loss))

        log['train_loss'].append(Train_loss)
        log['lr'].append(opt.param_groups[0]['lr'])
        # 学习率更新
        scheduler.step(i)
        # 验证
        model.eval()
        torch.no_grad()
        if not i % 1:
            for step, (x, y) in enumerate(valid_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                lossT = calc_loss(output, y)
                dice += calc_dice(output, y).item() * x.size(0)
                valid_loss += lossT.item() * x.size(0)
                if step == 0:
                    # 可视化
                    logit = torch.softmax(output, dim=1).detach().cpu()  # detach 拷贝，共享内存，脱离计算图， clone不共享，仍在计算图内
                    temp = logit[0].clone()
                    temp[0, :, :] = logit[0][2, :, :]
                    temp[1, :, :] = logit[0][0, :, :]
                    temp[2, :, :] = logit[0][1, :, :]
                    '''logit_img = torchvision.transforms.ToPILImage()(temp)
                    logit_img.save(result_path + '/Process/Epoch_' + str(i + 1) + '_loss_'
                                   + str(round(float(valid_loss / Batch_Size), 4)) + '.tif')'''

            Valid_loss = valid_loss / len(valid_idx)
            dice_values = dice / len(valid_idx)
            print('Validation Loss: {:.6f} \t Dice:{:.6f}'.format(Valid_loss, dice_values))
            log['valid_loss'].append(Valid_loss)
            log['dice_values'].append(dice_values)
            """ 模型存储 """
            if Valid_loss <= valid_loss_min:
                print(
                    'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, Valid_loss))
                torch.save(model.state_dict(), result_path + '/CheckPoint/' +'best.pkl')
                if Valid_loss <= valid_loss_min:
                    valid_loss_min = Valid_loss

    import pandas as pd
    df = pd.DataFrame.from_dict(log)
    df.to_excel(os.path.join(result_path, 'log.xlsx'))
if __name__ == '__main__':

    """ 超参数设置 """
    # writer = SummaryWriter('./train_log')
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_path', type=str, default='../../datasets/Data66/OCTA/Data_fold_1',
                        help="OCTA Data_path ")
    parser.add_argument('--cnn_Result_path', type=str, default='./Result',
                        help="OCTA Result_path ")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size used for training, validation and test")
    parser.add_argument('--lr', type=int, default=0.01,
                        help="initial_learning_rate")
    parser.add_argument('--cnn_epochs', type=int, default=30,
                        help="number of training epochs")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="number of dataloader thread")
    parser.add_argument('--pin_memory', type=int, default=4,
                        help="pin_memory")
    args = parser.parse_args()
    main(args)




