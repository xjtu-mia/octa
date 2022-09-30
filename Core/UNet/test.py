# Data_test
# Checkpoint
import os
from glob import glob
from natsort import natsorted
from PIL import Image, ImageChops, ImageOps
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.append(sys.path[0]+'/Core/UNet')
from Model.Unet_Family import U_Net,NestedUNet
import json
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torchsummary

import argparse

# GPU 检查GPU是否可用
test_on_gpu = torch.cuda.is_available()
if not test_on_gpu:
    print('Cuda is not available. Testing on CPU')
else:
    print('Cuda is available. Testing on GPU')
device = torch.device("cuda:0" if test_on_gpu else "cpu")
# device = torch.device("cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
def main(args):
    """　参数选择与设置　"""
    Result_path = args.cnn_Result_path
    Data_path = args.Data_path
    checkpoint_path = os.path.join(Result_path, 'CheckPoint')
    checkpoint_list = glob(os.path.join(checkpoint_path, '*'))
    checkpoint_list = natsorted(checkpoint_list)
    checkpoint = checkpoint_path + '/best.pkl'
    if args.test_all:
        save_path = os.path.join(Result_path, 'Test_Predict_all')
    else:
        save_path = os.path.join(Result_path, 'Test_Predict')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # U-Net 模型选择与设置(Input_channel and Output_channel)
    Model = {'deeplabV3pp': smp.DeepLabV3Plus}
    In_ch = 4
    Out_ch = 3
    # model = Model['AttU_Net'](In_ch, Out_ch)
    '''model = Model['deeplabV3pp'](in_channels=In_ch, 
                                classes=Out_ch,
                                activation=None,
                                encoder_weights=None,
                                )'''
    model = U_Net(in_ch=In_ch, out_ch=Out_ch)
    model.to(device)

    """ 加载模型参数 """
    model.load_state_dict(torch.load(checkpoint))
    model.eval()  # model.eval() ：不启用 BatchNormalization 和 Dropout


    feature_save_path = os.path.join(Result_path, 'feature_visual')
    if args.test_all:
        test_path = os.path.join(Data_path, 'test_all')  # 数据路径test
        label_path = os.path.join(Data_path, 'test_all_GT')  # test_GT
    else:
        test_path = os.path.join(Data_path, 'test')  # 数据路径test
        label_path = os.path.join(Data_path, 'test_GT')#test_GT
    label_images = natsorted(glob(os.path.join(label_path, "*.tif")))
    # 特征可视化
    module_name = []
    features_in_hook = []
    features_out_hook = []


    def hook(module, feat_in, feat_out):
        # print('hook working now')
        module_name.append(module.__class__)
        features_in_hook.append(feat_in)
        features_out_hook.append(feat_out)
        return None


    for child in model.children():
        if not isinstance(child, nn.MaxPool2d):
            child.register_forward_hook(hook=hook)

    # print(module_name)
    # print(features_in_hook)
    # print(features_out_hook)

    #这里的features_out_hook存储了所有输出的特征图
    def draw_feature(features_out_hook, features_save_path):
        for l_idx, layer in enumerate(features_out_hook):
            # print(layer[0].shape)
            if layer[0].shape[0] != 256:
                continue
            #扩展到5D然后压缩一个维度
            last_fea = F.interpolate(torch.unsqueeze(layer, 0), mode='trilinear', align_corners=True,size=[32,320, 320])
            last_fea = torch.squeeze(last_fea, 0)

            width = 320*8
            high = 320*4
            #之前的部分，这里注释掉
            # width = int(8*320/(layer[0].shape[0]/(2**5)))
            # high = 4*320
            # if l_idx == len(features_out_hook)-1:#最后一个特征图
            #     width = 3 * 320
            #     high = 320
            fv_img_all = Image.new('L', (width, high))
            fv_img_size = last_fea[0].shape[2]
            left = 0
            right = 0
            for c_idx, channel in enumerate(last_fea[0]):
                # print(channel.shape)
                fv_img = torchvision.transforms.ToPILImage()(channel.cpu())
                fv_img_all.paste(fv_img, (left, right, left+fv_img_size, right+fv_img_size))
                left = left + fv_img_size
                if left == width:
                    left = 0
                    right = right + fv_img_size
                # fv_img.save('Result_U-Net-424/feature_visual/'+str(l_idx)+'_'+str(c_idx)+'.tif')
            fv_img_all.save(os.path.join(features_save_path, img_name[:-4]+'_layer' + str(l_idx) + '.tif'))
        return 0


    # 预测模块
    def predict_img(input_img):
        img_A = Image.open(input_img[0]).convert('L')
        img_B = Image.open(input_img[1]).convert('L')
        img_C = Image.open(input_img[2]).convert('L')
        img_D = Image.open(input_img[3]).convert('L')
        img_A = torchvision.transforms.ToTensor()(img_A)
        img_B = torchvision.transforms.ToTensor()(img_B)
        img_C = torchvision.transforms.ToTensor()(img_C)
        img_D = torchvision.transforms.ToTensor()(img_D)
        img = torch.cat((img_A, img_B, img_C, img_D), dim=0)
        # img = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        img = img.unsqueeze(0).to(device)
        predict = model(img)

        logit = torch.softmax(predict, dim=1).detach().cpu()
        predict = logit.max(1)[1].unsqueeze(1).float()
        temp = logit[0].clone()
        temp[0, :, :] = logit[0][2, :, :]
        temp[1, :, :] = logit[0][0, :, :]
        temp[2, :, :] = logit[0][1, :, :]
        logit_img = torchvision.transforms.ToPILImage()(temp)
        predict[predict == 1] = 127 / 255
        predict[predict == 2] = 1
        predict_pil = torchvision.transforms.ToPILImage()(predict[0][0])
        return predict, logit_img, predict_pil



    test_A_path = os.path.join(test_path, 'data_A')
    test_B_path = os.path.join(test_path, 'data_B')
    test_C_path = os.path.join(test_path, 'data_C')
    test_D_path = os.path.join(test_path, 'data_D')

    test_A_images = natsorted(glob(os.path.join(test_A_path, "NDM[0-9][0-9][0-9][0-9]_?.tif")))
    test_B_images = natsorted(glob(os.path.join(test_B_path, "NDM[0-9][0-9][0-9][0-9]_?.tif")))
    test_C_images = natsorted(glob(os.path.join(test_C_path, "NDM[0-9][0-9][0-9][0-9]_?.tif")))
    test_D_images = natsorted(glob(os.path.join(test_D_path, "NDM[0-9][0-9][0-9][0-9]_?.tif")))
    test_images = zip(test_A_images, test_B_images, test_C_images, test_D_images)

    # test_crop_images = glob(os.path.join(test_path, "*c20.tif"))
    test_fliph_A_images = natsorted(glob(os.path.join(test_A_path, "*h.tif")))
    test_fliph_B_images = natsorted(glob(os.path.join(test_B_path, "*h.tif")))
    test_fliph_C_images = natsorted(glob(os.path.join(test_C_path, "*h.tif")))
    test_fliph_D_images = natsorted(glob(os.path.join(test_D_path, "*h.tif")))
    test_fliph_images = zip(test_fliph_A_images, test_fliph_B_images, test_fliph_C_images, test_fliph_D_images)

    test_flipv_A_images = natsorted(glob(os.path.join(test_A_path, "NDM[0-9][0-9][0-9][0-9]_?v.tif")))
    test_flipv_B_images = natsorted(glob(os.path.join(test_B_path, "NDM[0-9][0-9][0-9][0-9]_?v.tif")))
    test_flipv_C_images = natsorted(glob(os.path.join(test_C_path, "NDM[0-9][0-9][0-9][0-9]_?v.tif")))
    test_flipv_D_images = natsorted(glob(os.path.join(test_D_path, "NDM[0-9][0-9][0-9][0-9]_?v.tif")))
    test_flipv_images = zip(test_flipv_A_images, test_flipv_B_images, test_flipv_C_images, test_flipv_D_images)

    test_fliphv_A_images = natsorted(glob(os.path.join(test_A_path, "*hv.tif")))
    test_fliphv_B_images = natsorted(glob(os.path.join(test_B_path, "*hv.tif")))
    test_fliphv_C_images = natsorted(glob(os.path.join(test_C_path, "*hv.tif")))
    test_fliphv_D_images = natsorted(glob(os.path.join(test_D_path, "*hv.tif")))
    test_fliphv_images = zip(test_fliphv_A_images, test_fliphv_B_images, test_fliphv_C_images, test_fliphv_D_images)

    test_r90_A_images = natsorted(glob(os.path.join(test_A_path, "*r90.tif")))
    test_r90_B_images = natsorted(glob(os.path.join(test_B_path, "*r90.tif")))
    test_r90_C_images = natsorted(glob(os.path.join(test_C_path, "*r90.tif")))
    test_r90_D_images = natsorted(glob(os.path.join(test_D_path, "*r90.tif")))
    test_r90_images = zip(test_r90_A_images, test_r90_B_images, test_r90_C_images, test_r90_D_images)

    test_r270_A_images = natsorted(glob(os.path.join(test_A_path, "*r270.tif")))
    test_r270_B_images = natsorted(glob(os.path.join(test_B_path, "*r270.tif")))
    test_r270_C_images = natsorted(glob(os.path.join(test_C_path, "*r270.tif")))
    test_r270_D_images = natsorted(glob(os.path.join(test_D_path, "*r270.tif")))
    test_r270_images = zip(test_r270_A_images, test_r270_B_images, test_r270_C_images, test_r270_D_images)

    print(f'Test data number is {len(test_A_images)};')

    Input_Data = zip(test_images, test_fliph_images, test_flipv_images, test_fliphv_images, test_r90_images,
                     test_r270_images, label_images)

    test_metric = dict()
    for test_img, h_img, v_img, hv_img, r90_img, r270_img, label_img in tqdm(Input_Data):
        img_name = os.path.basename(test_img[0])
        module_name = []
        features_in_hook = []
        features_out_hook = []
        o_pre, o_prob, o_img = predict_img(test_img)  # pre 预测的概率图， img预测的结果图 经过max
        draw_feature(features_out_hook, feature_save_path)
        #缩放一下
        # layer = features_out_hook[1]
        # layer = F.interpolate(torch.unsqueeze(layer, 0), mode='trilinear', align_corners=True,size=[32,320, 320])
        # layer = torch.squeeze(layer, 0)

        layer = features_out_hook[1]
        layer = F.interpolate(torch.unsqueeze(layer, 0), mode='trilinear', align_corners=True,size=[32,320, 320])
        layer = torch.squeeze(layer, 0)

        features = layer[0].detach().cpu()
        features = features.transpose(0, 1).transpose(1, 2).contiguous()

        h_pre, h_prob, h_img = predict_img(h_img)
        v_pre, v_prob, v_img = predict_img(v_img)
        hv_pre, hv_prob, hv_img = predict_img(hv_img)
        r90_pre, r90_prob, r90_img = predict_img(r90_img)
        r270_pre, r270_prob, r270_img = predict_img(r270_img)

        o_result = o_img
        h_result = h_img.transpose(Image.FLIP_LEFT_RIGHT)
        v_result = v_img.transpose(Image.FLIP_TOP_BOTTOM)
        hv_result = hv_img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
        r90_result = r90_img.rotate(360 - 90, resample=Image.NEAREST)
        r270_result = r270_img.rotate(360 - 270, resample=Image.NEAREST)

        # result 存储
        o_result.save(os.path.join(save_path, img_name))
        h_result.save(os.path.join(save_path, img_name[:-4] + 'h.tif'))
        v_result.save(os.path.join(save_path, img_name[:-4] + 'v.tif'))
        hv_result.save(os.path.join(save_path, img_name[:-4] + 'hv.tif'))
        r90_result.save(os.path.join(save_path, img_name[:-4] + 'r90.tif'))
        r270_result.save(os.path.join(save_path, img_name[:-4] + 'r270.tif'))

        h_prob = h_prob.transpose(Image.FLIP_LEFT_RIGHT)
        v_prob = v_prob.transpose(Image.FLIP_TOP_BOTTOM)
        hv_prob = hv_prob.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
        r90_prob = r90_prob.rotate(360 - 90, resample=Image.NEAREST)
        r270_prob = r270_prob.rotate(360 - 270, resample=Image.NEAREST)

        o_prob.save(os.path.join(save_path, img_name[:-4] + 'x.tif'))
        h_prob.save(os.path.join(save_path, img_name[:-4] + 'xh.tif'))
        v_prob.save(os.path.join(save_path, img_name[:-4] + 'xv.tif'))
        hv_prob.save(os.path.join(save_path, img_name[:-4] + 'xhv.tif'))
        r90_prob.save(os.path.join(save_path, img_name[:-4] + 'xr90.tif'))
        r270_prob.save(os.path.join(save_path, img_name[:-4] + 'xr270.tif'))

        o_prob = np.array(o_prob).astype(np.int64)
        h_prob = np.array(h_prob).astype(np.int64)
        v_prob = np.array(v_prob).astype(np.int64)
        hv_prob = np.array(hv_prob).astype(np.int64)
        r90_prob = np.array(r90_prob).astype(np.int64)
        r270_prob = np.array(r270_prob).astype(np.int64)

        # 分割结果 0 动脉 2 静脉 1是背景
        sum_prob = o_prob + h_prob + v_prob + hv_prob + r90_prob + r270_prob
        # sum_prob = o_prob

        sum_result = sum_prob.argmax(2)
        pre_a = np.zeros_like(sum_result)
        pre_v = np.zeros_like(sum_result)
        pre_vessel = np.zeros_like(sum_result)
        pre_a[sum_result == 0] = 1   # 注意 这里 0维为动脉,是为了后面彩色图像输出
        pre_v[sum_result == 2] = 1   # 2维为静脉
        pre_vessel[sum_result != 1] = 1

        label = Image.open(label_img)
        gt = np.array(label)
        gt_a = np.zeros_like(gt)
        gt_v = np.zeros_like(gt)
        gt_vessel = np.zeros_like(gt)
        gt_a[gt == 255] = 1
        gt_v[gt == 127] = 1
        gt_vessel = gt_a + gt_v
        gt_vessel[gt_vessel > 1] = 1

        dice_a = 2 * np.sum(pre_a*gt_a) / (np.sum(pre_a) + np.sum(gt_a))
        dice_v = 2 * np.sum(pre_v * gt_v) / (np.sum(pre_v) + np.sum(gt_v))
        dice_vessel = 2 * np.sum(pre_vessel * gt_vessel) / (np.sum(pre_vessel) + np.sum(gt_vessel))
        tp_vessel = np.sum(pre_vessel * gt_vessel)
        tn_vessel = np.sum((np.ones_like(pre_vessel)-pre_vessel) * (np.ones_like(gt_vessel)-gt_vessel))
        fp_vessel = np.sum(pre_vessel * (np.ones_like(gt_vessel)-gt_vessel))
        fn_vessel = np.sum((np.ones_like(pre_vessel)-pre_vessel) * gt_vessel)
        recall = tp_vessel / (tp_vessel + fn_vessel)
        specificity = tn_vessel / (fp_vessel + tn_vessel)
        # test_dice.append([dice_a, dice_v, dice_vessel])

        Union_vessel = pre_vessel + gt_vessel
        Union_vessel[Union_vessel > 1] = 1
        Os = np.sum(Union_vessel) - np.sum(gt_vessel)
        OR = Os/np.sum(Union_vessel)
        Us = np.sum(Union_vessel) - np.sum(pre_vessel)
        UR = Us/np.sum(Union_vessel)
        print(img_name, dice_a, dice_v, dice_vessel, OR, UR)

        test_dice = dict()
        test_dice['artery_dice'] = dice_a
        test_dice['vein_dice'] = dice_v
        test_dice['vessel_dice'] = dice_vessel
        test_dice['recall'] = recall
        test_dice['specificity'] = specificity
        test_dice['OR'] = OR
        test_dice['UR'] = UR
        test_metric[img_name] = test_dice

        sum_result[sum_result == 0] = 255
        sum_result[sum_result == 1] = 0
        sum_result[sum_result == 2] = 127
        sum_result = sum_result.astype(np.uint8)
        sum_img = Image.fromarray(sum_result)
        sum_img.save(os.path.join(save_path, img_name[:-4] + 'seg.tif'))
        sum_result_rgb = np.zeros((320, 320, 3), np.uint8)
        sum_result_rgb[:, :, 0][sum_result == 255] = 255
        sum_result_rgb[:, :, 2][sum_result == 127] = 255
        sum_img_rgb = Image.fromarray(sum_result_rgb)
        sum_img_rgb.save(os.path.join(save_path, img_name[:-4] + 'seg_rgb.tif'))

        mean_result = (o_prob + h_prob + v_prob + hv_prob + r90_prob + r270_prob) / 6
        mean_result = mean_result.astype(np.uint8)
        mean_result_gray = mean_result[:, :, 0] + mean_result[:, :, 2]//2
        mean_img = Image.fromarray(mean_result)
        mean_img.save(os.path.join(save_path, img_name[:-4] + 'seg_mean_prob.tif'))
        mean_result_gray = Image.fromarray(mean_result_gray)
        diff_img = ImageChops.difference(label, mean_result_gray)
        diff_img = ImageOps.invert(diff_img)
        diff_img.save(os.path.join(save_path, img_name[:-4] + 'seg_mean_prob_diff.tif'))

        mean_img_gray = Image.fromarray(mean_result[:, :, 0]+mean_result[:, :, 2])
        mean_img_gray.save(os.path.join(save_path, img_name[:-4] + 'seg_vessel_prob.tif'))

        # 预测结果的概率特征，第一维度 动脉 第二维度 背景 第三维度是静脉
        init_feature = np.concatenate(((o_prob/255+h_prob/255+v_prob/255+hv_prob/255+r90_prob/255+r270_prob/255)/6, features), axis=2)
        np.save(os.path.join(save_path, img_name[:-4] + 'Prob.npy'), init_feature)

    test_metric_str = json.dumps(test_metric)

    with open(os.path.join(Result_path, 'test_metric.json'), 'w') as f:
        f.write(test_metric_str)

    mean = np.zeros((len(test_metric.values()), 2))
    for i, v in enumerate(test_metric.values()):
        mean[i, 0] = v['recall']
        mean[i, 1] += v['specificity']
    std = np.std(mean, axis=0)
    mean = np.mean(mean, axis=0)

    '''import pandas as pd
    df = pd.DataFrame.from_dict(test_metric)
    df.to_excel(os.path.join(Result_path, '实验结果.xlsx'))
    df = pd.DataFrame.from_dict({'recall': [mean[0], std[0]], 'specificity': [mean[1], std[1]]})
    df.to_excel(os.path.join(Result_path, '实验结果_final.xlsx'))'''
if __name__ == '__main__':
    """ 加载模型参数 """

    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_path', type=str, default='../../datasets/Data66/OCTA/Data_fold_1',
                        help="OCTA Data_path ")
    parser.add_argument('--Result_path', type=str, default='./Result',
                        help="OCTA Result_path ")
    parser.add_argument('--test_all', type=int, default=0,
                        help="0: testing on test set, 1: testing on all images")
    args = parser.parse_args()
    main(args)